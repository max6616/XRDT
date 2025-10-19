#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All-in-one simulator and canonicalizer.

This script first generates the dataset using the existing simulator, then
canonicalizes HKL labels of the generated samples into the ASU using cctbx.

CLI exposes simulator options and canonicalization options in one place.
"""

import os
import json
import argparse
import multiprocessing
import glob
import random
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Import simulator entry point (same directory module)
from simulator import main as simulator_main
from simulator import process_single_cif, init_worker, save_single_sample

# cctbx for canonicalization
from cctbx import crystal, miller
from cctbx.array_family import flex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset and canonicalize HKL labels in one run"
    )

    # Simulator options (kept compatible with simulator.py)
    parser.add_argument('--cif_path', type=str, default='/media/max/Data/datasets/mp_all')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--debug', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--angle_step', type=float, default=0.5)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.01)
    parser.add_argument('--min_sequence_length', type=int, default=1024)
    parser.add_argument('--max_sequence_length', type=int, default=32768)

    # Device parameters
    parser.add_argument('--detector_size', type=tuple, default=(2560, 2560))
    parser.add_argument('--detector_dist', type=tuple, default=(500, 500))
    parser.add_argument('--detector_poni', type=tuple, default=(1280, 1280, 1280, 1280))
    parser.add_argument('--xray_wavelength', type=float, default=0.413)
    parser.add_argument('--xray_bandwidth', type=float, default=0.02)

    # Additional crystal parameters to record from CIF files
    parser.add_argument('--extra_params', default=[
        '_symmetry_Int_Tables_number',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
    ], type=list)

    # Canonicalization options
    parser.add_argument('--no-canonicalize', dest='do_canonicalize', action='store_false',
                        help='Disable the canonicalization step')
    parser.set_defaults(do_canonicalize=True)
    parser.add_argument('--no-inline', dest='inline', action='store_false',
                        help='Disable inline canonicalization; use two-pass mode')
    parser.set_defaults(inline=True)
    parser.add_argument('--canon_output_dir', type=str, default='/media/max/Data/datasets/all_in_one_test',
                        help='Output root directory for canonicalized dataset. '
                             'Default: save_path + "_canonical"')
    parser.add_argument('--canon_ext', type=str, default='.jsonl',
                        help='File extension to process during canonicalization')
    parser.add_argument('--canon_processes', type=int, default=32,
                        help='Number of worker processes for canonicalization; use 1 for sequential')
    parser.add_argument('--canon_chunksize', type=int, default=1,
                        help='Chunksize for imap_unordered in canonicalization')
    parser.add_argument('--canon_report_width', type=int, default=50,
                        help='Width for canonicalization report separators')

    return parser.parse_args()


# -------------------- Canonicalization helpers (from canonicalize.py) --------------------
def canonicalize_hkl_cctbx(hkl_tuple, space_group_number):
    """Map input HKL to the ASU using cctbx miller set mapping."""
    symm = crystal.symmetry(space_group_symbol=str(space_group_number))
    miller_set = miller.set(
        crystal_symmetry=symm,
        indices=flex.miller_index([hkl_tuple]),
        anomalous_flag=False,
    )
    miller_set_asu = miller_set.map_to_asu()
    canonical_hkl = miller_set_asu.indices()[0]
    return list(canonical_hkl)


def process_sample(sample_line):
    sample = json.loads(sample_line)
    sg_number = int(sample['metadata']['crystal_params']['_symmetry_Int_Tables_number'])

    original_labels = sample['labels']
    canonical_labels = [canonicalize_hkl_cctbx(tuple(hkl), sg_number)
                        for hkl in original_labels]

    changed_count = 0
    total_labels = len(original_labels)

    for orig, canon in zip(original_labels, canonical_labels):
        if orig != canon:
            changed_count += 1

    sample['labels'] = canonical_labels
    sample['label_change_stats'] = {
        'total_labels': total_labels,
        'changed_labels': changed_count,
        'change_ratio': changed_count / total_labels if total_labels > 0 else 0,
    }

    return json.dumps(sample)


def run_processing(file_paths):
    input_path, output_path = file_paths
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_stats = {
        'total_samples': 0,
        'total_labels': 0,
        'changed_labels': 0,
        'samples_with_changes': 0,
    }

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            result = process_sample(line)
            if result:
                sample = json.loads(result)
                stats = sample['label_change_stats']

                file_stats['total_samples'] += 1
                file_stats['total_labels'] += stats['total_labels']
                file_stats['changed_labels'] += stats['changed_labels']
                if stats['changed_labels'] > 0:
                    file_stats['samples_with_changes'] += 1

                outfile.write(result + '\n')

    return file_stats


def get_all_filepaths(input_root_dir, output_root_dir, file_extension):
    filepaths = []
    for dir_name in os.listdir(input_root_dir):
        input_dir = os.path.join(input_root_dir, dir_name)
        if os.path.isdir(input_dir):
            for filename in os.listdir(input_dir):
                if filename.endswith(file_extension):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_root_dir, dir_name, filename)
                    filepaths.append((input_path, output_path))
    return filepaths


def run_canonicalization(input_root_dir, output_root_dir, ext, processes, chunksize, report_width):
    print("\nStarting label canonicalization...")
    print(f"Input path: {input_root_dir}")
    print(f"Output path: {output_root_dir}")

    all_files = get_all_filepaths(input_root_dir, output_root_dir, ext)
    if not all_files:
        print(f"No {ext} files found to process for canonicalization.")
        return

    global_stats = {
        'total_samples': 0,
        'total_labels': 0,
        'changed_labels': 0,
        'samples_with_changes': 0,
    }

    if processes <= 1:
        results = []
        for fp in tqdm(all_files, total=len(all_files), desc="Canonicalization Progress"):
            results.append(run_processing(fp))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(run_processing, all_files, chunksize=chunksize),
                total=len(all_files), desc="Canonicalization Progress"
            ))

    for file_stats in results:
        global_stats['total_samples'] += file_stats['total_samples']
        global_stats['total_labels'] += file_stats['total_labels']
        global_stats['changed_labels'] += file_stats['changed_labels']
        global_stats['samples_with_changes'] += file_stats['samples_with_changes']

    print("\n" + "=" * report_width)
    print("Label Change Statistics:")
    print("=" * report_width)
    print(f"Total samples: {global_stats['total_samples']:,}")
    print(f"Total labels: {global_stats['total_labels']:,}")
    print(f"Changed labels: {global_stats['changed_labels']:,}")
    print(f"Samples with changed labels: {global_stats['samples_with_changes']:,}")

    if global_stats['total_labels'] > 0:
        label_change_ratio = global_stats['changed_labels'] / global_stats['total_labels']
        sample_change_ratio = (global_stats['samples_with_changes'] /
                               global_stats['total_samples'] if global_stats['total_samples'] > 0 else 0)
        print(f"Label change ratio: {label_change_ratio:.2%}")
        print(f"Sample change ratio: {sample_change_ratio:.2%}")

    print("=" * report_width)
    print("Canonicalization finished successfully!")


def main(cli_args):
    # If inline canonicalization is enabled and canonicalization is desired, run inline pipeline
    if getattr(cli_args, 'inline', True) and getattr(cli_args, 'do_canonicalize', True):
        return main_inline(cli_args)

    # Fallback to two-pass mode (simulate then canonicalize)
    print("Starting dataset simulation...")
    simulator_main(cli_args)
    print("Dataset simulation finished.")

    if not getattr(cli_args, 'do_canonicalize', True):
        print("Canonicalization step disabled. Exiting.")
        return

    input_root_dir = cli_args.save_path
    output_root_dir = (cli_args.canon_output_dir
                       if cli_args.canon_output_dir is not None
                       else f"{cli_args.save_path}_canonical")

    run_canonicalization(
        input_root_dir=input_root_dir,
        output_root_dir=output_root_dir,
        ext=cli_args.canon_ext,
        processes=cli_args.canon_processes,
        chunksize=cli_args.canon_chunksize,
        report_width=cli_args.canon_report_width,
    )


# -------------------- Inline canonicalization pipeline --------------------
def canonicalize_sample_inplace(sample):
    """Canonicalize HKL labels in-place and return change stats."""
    try:
        sg_number = int(sample['metadata']['crystal_params']['_symmetry_Int_Tables_number'])
    except Exception:
        return {'total_labels': len(sample.get('labels', [])), 'changed_labels': 0, 'change_ratio': 0}

    original_labels = sample['labels']
    canonical_labels = [canonicalize_hkl_cctbx(tuple(hkl), sg_number) for hkl in original_labels]

    changed_count = sum(1 for orig, canon in zip(original_labels, canonical_labels) if orig != canon)

    sample['labels'] = canonical_labels
    stats = {
        'total_labels': len(original_labels),
        'changed_labels': changed_count,
        'change_ratio': changed_count / len(original_labels) if len(original_labels) > 0 else 0,
    }
    sample['label_change_stats'] = stats
    return stats


def process_and_save_and_count_inline(cif_file, args, output_dir, split_name, counter, lock):
    """Worker: simulate one CIF, canonicalize labels, then save."""
    try:
        result = process_single_cif(args, cif_file)
        if result is not None:
            canonicalize_sample_inplace(result)
            with lock:
                current_index = counter.value
                counter.value += 1
            save_single_sample(result, output_dir, split_name, current_index)
            return True
    except Exception as e:
        # Keep logs concise in large-scale processing
        tqdm.write(f"A task failed for {os.path.basename(cif_file)}: {e}")
    return False


def process_and_save_split_inline(args, cif_files, split_name, output_root_dir):
    if not cif_files:
        print(f"No files to process for split: {split_name}")
        return 0

    output_dir = os.path.join(output_root_dir, split_name)
    os.makedirs(output_dir, exist_ok=True)

    CHUNK_SIZE = 10000

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    print(f"\nProcessing {len(cif_files)} files for '{split_name}' split in chunks of {CHUNK_SIZE}...")
    pbar = tqdm(total=len(cif_files), desc=f"Processing {split_name}", dynamic_ncols=True)

    for i in range(0, len(cif_files), CHUNK_SIZE):
        chunk = cif_files[i:i + CHUNK_SIZE]
        pbar.set_description(f"Processing {split_name} (chunk {i//CHUNK_SIZE + 1})")

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker) as executor:
            task_func = partial(process_and_save_and_count_inline, args=args, output_dir=output_dir,
                                split_name=split_name, counter=counter, lock=lock)
            results_iterator = executor.map(task_func, chunk)
            for _ in results_iterator:
                pbar.update(1)

    pbar.close()
    final_count = counter.value
    print(f"Successfully generated {final_count} samples for '{split_name}' split.")
    return final_count


def main_inline(args):
    start_time = time.time()

    # Enumerate CIF files
    cif_files = glob.glob(f"{args.cif_path}/*.cif")
    if args.debug and args.debug > 0:
        print(f"Debug mode, randomly selecting {args.debug} CIF files")
        cif_files = random.sample(cif_files, min(args.debug, len(cif_files)))

    if not cif_files:
        print("No CIF files found. Exiting.")
        return

    print(f"Found {len(cif_files)} total CIF files.")

    # Decide output root dir for canonicalized dataset
    output_root_dir = (args.canon_output_dir
                       if args.canon_output_dir is not None
                       else (f"{args.save_path}_canonical" if args.save_path else None))
    if output_root_dir is None:
        raise ValueError("Output directory is not specified. Set --canon_output_dir or --save_path.")

    os.makedirs(output_root_dir, exist_ok=True)
    with open(os.path.join(output_root_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Shuffle and split by files
    print("Shuffling and splitting CIF file list...")
    random.shuffle(cif_files)
    total_files = len(cif_files)
    test_size = int(total_files * args.test_ratio)
    val_size = int(total_files * args.val_ratio)
    test_cifs = cif_files[:test_size]
    val_cifs = cif_files[test_size:test_size + val_size]
    train_cifs = cif_files[test_size + val_size:]

    print(f"Dataset split (by files): Training {len(train_cifs)}, Val {len(val_cifs)}, Test {len(test_cifs)}")

    # Process each split with inline canonicalization
    train_count = process_and_save_split_inline(args, train_cifs, 'train', output_root_dir)
    val_count = process_and_save_split_inline(args, val_cifs, 'val', output_root_dir)
    test_count = process_and_save_split_inline(args, test_cifs, 'test', output_root_dir)

    total_samples_count = train_count + val_count + test_count

    # Final statistics
    print("\n" + "=" * 50)
    print("Dataset Creation Summary (inline canonicalization)")
    print("=" * 50)
    print(f"Total CIF files processed: {total_files}")
    print(f"Total samples successfully generated: {total_samples_count}")
    print(f"  - Training samples:   {train_count}")
    print(f"  - Val samples: {val_count}")
    print(f"  - Test samples:       {test_count}")

    stats = {
        "total_cifs_found": total_files,
        "total_samples_generated": total_samples_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "processing_time_seconds": time.time() - start_time,
        "mode": "inline",
    }

    with open(os.path.join(output_root_dir, "dataset_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDataset creation complete! Total time: {stats['processing_time_seconds']:.1f} seconds")
    print(f"Dataset saved to: {output_root_dir}")
    print("Each sample is saved as an individual .jsonl file in its respective split folder.")


if __name__ == '__main__':
    args = parse_args()
    main(args)


