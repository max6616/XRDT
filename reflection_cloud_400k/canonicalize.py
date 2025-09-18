#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
import multiprocessing
from cctbx import crystal, miller
from cctbx.array_family import flex


def parse_args():
    parser = argparse.ArgumentParser(description="Canonicalize HKL labels to ASU using cctbx.")
    parser.add_argument('-i', '--input-dir', default='/media/max/Data/datasets/mp_random_150k_v4', help='Input dataset root directory')
    parser.add_argument('-o', '--output-dir', default='/media/max/Data/datasets/mp_random_150k_v4_canonical_tmp', help='Output dataset root directory')
    parser.add_argument('--ext', default='.jsonl', help='File extension to process')
    parser.add_argument('-p', '--processes', type=int, default=os.cpu_count(), help='Number of worker processes; use 1 for sequential')
    parser.add_argument('--chunksize', type=int, default=1, help='Chunksize for imap_unordered')
    parser.add_argument('--report-width', type=int, default=50, help='Width for report separators')
    return parser.parse_args()

def canonicalize_hkl_cctbx(hkl_tuple, space_group_number):
    symm = crystal.symmetry(space_group_symbol=str(space_group_number))
    
    miller_set = miller.set(
        crystal_symmetry=symm,
        indices=flex.miller_index([hkl_tuple]),
        anomalous_flag=False)
    
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
    
    for i, (orig, canon) in enumerate(zip(original_labels, canonical_labels)):
        if orig != canon:
            changed_count += 1
    
    sample['labels'] = canonical_labels
    sample['label_change_stats'] = {
        'total_labels': total_labels,
        'changed_labels': changed_count,
        'change_ratio': changed_count / total_labels if total_labels > 0 else 0
    }
    
    return json.dumps(sample)

def run_processing(file_paths):
    input_path, output_path = file_paths
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    file_stats = {
        'total_samples': 0,
        'total_labels': 0,
        'changed_labels': 0,
        'samples_with_changes': 0
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

def main():
    args = parse_args()
    print("Starting dataset canonicalization...")
    print(f"Input path: {args.input_dir}")
    print(f"Output path: {args.output_dir}")

    all_files = get_all_filepaths(args.input_dir, args.output_dir, args.ext)
    if not all_files:
        print(f"No {args.ext} files found to process.")
        return

    global_stats = {
        'total_samples': 0,
        'total_labels': 0,
        'changed_labels': 0,
        'samples_with_changes': 0
    }
    
    if args.processes <= 1:
        results = []
        for fp in tqdm(all_files, total=len(all_files), desc="Overall Progress"):
            results.append(run_processing(fp))
    else:
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(run_processing, all_files, chunksize=args.chunksize),
                total=len(all_files), desc="Overall Progress"
            ))
    
    for file_stats in results:
        global_stats['total_samples'] += file_stats['total_samples']
        global_stats['total_labels'] += file_stats['total_labels']
        global_stats['changed_labels'] += file_stats['changed_labels']
        global_stats['samples_with_changes'] += file_stats['samples_with_changes']
    
    print("\n" + "="*args.report_width)
    print("Label Change Statistics:")
    print("="*args.report_width)
    print(f"Total samples: {global_stats['total_samples']:,}")
    print(f"Total labels: {global_stats['total_labels']:,}")
    print(f"Changed labels: {global_stats['changed_labels']:,}")
    print(f"Samples with changed labels: {global_stats['samples_with_changes']:,}")
    
    if global_stats['total_labels'] > 0:
        label_change_ratio = global_stats['changed_labels'] / global_stats['total_labels']
        sample_change_ratio = global_stats['samples_with_changes'] / global_stats['total_samples']
        print(f"Label change ratio: {label_change_ratio:.2%}")
        print(f"Sample change ratio: {sample_change_ratio:.2%}")
    
    print("="*args.report_width)
    print("Canonicalization finished successfully!")


if __name__ == '__main__':
    main()