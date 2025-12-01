from datad import UnitCell, SingleXtal, Xray, Pattern2D
from detector import Detector
import argparse
import numpy as np
from tqdm import tqdm
import os
import json
import glob
import random
import time
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from utils import reset_seed
import math
import gc
from functools import partial


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_path', type=str, default='/media/max/Data/datasets/mp_all')
    parser.add_argument('--save_path', type=str, default='/media/max/Data/datasets/mp_random_150k_test')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode, randomly sample n cif files')
    parser.add_argument('--num_workers', type=int, default=28, help='Number of parallel workers')
    parser.add_argument('--angle_step', type=float, default=0.5, help='Angle step (degrees)')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.01, help='Test set ratio')
    parser.add_argument('--min_sequence_length', type=int, default=1024, help='Minimum sequence length')
    parser.add_argument('--max_sequence_length', type=int, default=32768, help='Maximum sequence length')

    # Device parameters
    parser.add_argument('--detector_size', type=tuple, default=(2560, 2560), help='Detector size [sizex, sizey]')
    parser.add_argument('--detector_dist', type=tuple, default=(500, 500), help='Detector distance range [min, max]')
    parser.add_argument('--detector_poni', type=tuple, default=(1280, 1280, 1280, 1280), help='Detector center range [ponix_min, ponix_max, poniy_min, poniy_max]')
    parser.add_argument('--xray_wavelength', type=float, default=0.413, help='X-ray wavelength')
    parser.add_argument('--xray_bandwidth', type=float, default=0.02, help='X-ray bandwidth')
    
    # Additional crystal parameters to record
    parser.add_argument('--extra_params', default=[
        '_symmetry_Int_Tables_number',
        '_cell_angle_alpha',
        '_cell_angle_beta', 
        '_cell_angle_gamma',
        '_cell_length_a',
        '_cell_length_b', 
        '_cell_length_c',
    ], type=list, help='Additional parameters to record from CIF files')
    
    return parser.parse_args()


def init_worker():
    """Initialize random seed for worker process"""
    seed = (int(time.time() * 1000) + os.getpid() + threading.get_ident()) & 0xFFFFFFFF
    np.random.seed(seed)
    random.seed(seed + 1)


def random_detector_setup(args):
    """Generate random detector parameters"""
    reset_seed()
    
    detector_params = {
        'normal': (0, 0, 1),
        'vx': (1, 0, 0),
        'sizex': args.detector_size[0],
        'sizey': args.detector_size[1], 
        'dist': np.random.uniform(args.detector_dist[0], args.detector_dist[1]),
        'ponix': np.random.uniform(args.detector_poni[0], args.detector_poni[1]),
        'poniy': np.random.uniform(args.detector_poni[2], args.detector_poni[3]),
        'ps': 1
    }
    
    xray_params = {
        'mu': args.xray_wavelength,
        'sig': args.xray_wavelength * args.xray_bandwidth / 2.355
    }
    
    return detector_params, xray_params


def collect_diffraction_sequence(cif_file, detector_params, xray_params, angle_step=0.1):
    """
    Collect diffraction sequence data for complete 360 degree rotation
    """
    try:
        # Initialize crystal and device
        try:
            unitcell = UnitCell.from_cif(cif_file)
        except Exception as e:
            # print(f"Error reading CIF file {cif_file}: {str(e)}")
            return None, None, None, None
        xtal = SingleXtal.random(unitcell)
        # xtal = SingleXtal.from_rcp_vectors(unitcell, x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1))
        detector = Detector(**detector_params)
        xray = Xray.Gaussian(**xray_params)
        
        # Collect 360 degree data
        all_peaks = []
        angles = np.arange(0, 360, angle_step)
        
        for angle in angles:
            rotated_xtal = xtal.copy()
            rotated_xtal.rotate_by_axis_angle(axis=(0, 1, 0), angle=angle)
            
            p2d = Pattern2D(rotated_xtal, xray, inc=(0, 0, -1), vx=(1, 0, 0))
            p2d.calc_peaks(min_tth=1, max_tth=89, sort_by="intensity")
            
            if len(p2d.intns) > 0:
                detector.project_peaks(p2d)
                
                angle_peaks = []
                for i in range(p2d.num):
                    if detector._mask[i]:
                        peak_data = {
                            'angle': round(angle, 5),
                            'intensity': round(p2d.intns[i], 5),
                            'hkl': tuple(p2d.hkls[i].astype(int))
                        }
                        angle_peaks.append(peak_data)
                
                for i in range(len(angle_peaks)):
                    angle_peaks[i]['px'] = detector._px[i]
                    angle_peaks[i]['py'] = detector._py[i]
                
                all_peaks.extend(angle_peaks)
        
        return all_peaks, detector_params, unitcell, xtal
        
    except Exception as e:
        # Suppress verbose errors during multiprocessing
        print(f"Error processing file {cif_file}: {str(e)}")
        return None, None, None, None


def extract_cif_metadata(cif_file, extra_params):
    """Extract metadata from CIF file"""
    metadata = {}
    if os.path.exists(cif_file):
        try:
            with open(cif_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    for param in extra_params:
                        if line.startswith(param):
                            try:
                                metadata[param] = line.split()[1]
                            except:
                                metadata[param] = None
                            break
        except Exception as e:
            print(f"Error reading CIF file {cif_file}: {str(e)}")
    return metadata


def normalize_sequence_data(peaks_data, detector_params):
    """
    Normalize diffraction data to sequence format
    """
    if not peaks_data:
        return [], []
    
    input_sequence = []
    labels = []
    
    for peak in peaks_data:
        # if peak['intensity'] < 0.1:
        #     continue
        angle_norm = peak['angle'] / 360.0
        x_norm = peak['px'] / detector_params['sizex']
        y_norm = peak['py'] / detector_params['sizey']
        intensity = math.log1p(peak['intensity'])
        
        input_sequence.append([angle_norm, x_norm, y_norm, intensity])
        h, k, l = peak['hkl']
        labels.append([int(h), int(k), int(l)])
    
    return input_sequence, labels


def optimize_xrd_sequence(peaks_data, max_length=2048):
    """Optimize sequence for XRD data"""
    if len(peaks_data) <= max_length:
        return peaks_data
    
    sorted_peaks = sorted(peaks_data, key=lambda x: x['intensity'], reverse=True)
    selected_peaks = sorted_peaks[:max_length]
    selected_peaks.sort(key=lambda x: x['angle'])
    
    return selected_peaks


def process_single_cif(args, cif_file):
    """Process single CIF file"""
    try:
        detector_params, xray_params = random_detector_setup(args)
        
        peaks_data, final_detector_params, unitcell, xtal = collect_diffraction_sequence(
            cif_file, detector_params, xray_params, args.angle_step
        )
        
        if peaks_data is None or len(peaks_data) < args.min_sequence_length:
            return None
            
        if len(peaks_data) > args.max_sequence_length:
            peaks_data = optimize_xrd_sequence(peaks_data, args.max_sequence_length)
        
        input_sequence, labels = normalize_sequence_data(peaks_data, final_detector_params)
        input_sequence = [[round(x, 5) for x in peak] for peak in input_sequence]
        
        if len(input_sequence) == 0:
            return None
        
        cif_metadata = extract_cif_metadata(cif_file, args.extra_params)
        
        sample = {
            "input_sequence": input_sequence,
            "labels": labels,
            "metadata": {
                "cif_file": os.path.basename(cif_file),
                "rotation_matrix": xtal.rcp_matrix.tolist(),
                "crystal_params": cif_metadata,
            }
        }
        return sample
        
    except Exception as e:
        print(f"Error processing file {cif_file}: {str(e)}")
        return None


def save_single_sample(sample, output_dir, split_name, index):
    """Saves a single sample to a uniquely named .jsonl file."""
    filename = f"{split_name}_{index:06d}.jsonl"
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Failed to save sample to {filepath}: {e}")

def process_and_save_split(args, cif_files, split_name):
    """
    [已修正] 通过将任务分解为更小的“块”(chunks)来处理超大规模文件。
    为每个块创建一个新的进程池，并使用正确的Manager.Lock来保证计数器安全。
    """
    if not cif_files:
        print(f"No files to process for split: {split_name}")
        return 0

    output_dir = os.path.join(args.save_path, split_name)
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_SIZE = 10000
    
    # 使用 Manager 来创建跨进程共享的计数器和锁
    manager = mp.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()  # <-- 1. 创建一个正确的锁对象

    print(f"\nProcessing {len(cif_files)} files for '{split_name}' split in chunks of {CHUNK_SIZE}...")
    
    pbar = tqdm(total=len(cif_files), desc=f"Processing {split_name}", dynamic_ncols=True)
    
    for i in range(0, len(cif_files), CHUNK_SIZE):
        chunk = cif_files[i:i + CHUNK_SIZE]
        pbar.set_description(f"Processing {split_name} (chunk {i//CHUNK_SIZE + 1})")
        
        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker) as executor:
            # 2. 将锁(lock)也传递给工作函数
            task_func = partial(process_and_save_and_count, args=args, output_dir=output_dir, 
                                split_name=split_name, counter=counter, lock=lock)
            
            results_iterator = executor.map(task_func, chunk)
            
            for _ in results_iterator:
                pbar.update(1)
    
    pbar.close()

    final_count = counter.value
    print(f"Successfully generated {final_count} samples for '{split_name}' split.")
    return final_count

# --- 辅助函数 process_and_save_and_count 保持不变 ---
def process_and_save_and_count(cif_file, args, output_dir, split_name, counter):
    """
    这个辅助函数封装了处理和保存单个文件的逻辑，并更新计数器。
    """
    try:
        result = process_single_cif(args, cif_file)
        if result is not None:
            # get_lock() 确保多进程环境下计数器更新的线程安全
            with counter.get_lock():
                current_index = counter.value
                counter.value += 1
            save_single_sample(result, output_dir, split_name, current_index)
            return True # 表示成功
    except Exception as e:
        # 在大规模处理中，不建议打印过多信息，除非用于调试
        tqdm.write(f"A task failed for {os.path.basename(cif_file)}: {e}")
        pass
    return False # 表示失败

# --- 需要一个小的辅助函数来配合 partial 和 map ---
def process_and_save_and_count(cif_file, args, output_dir, split_name, counter, lock):
    """
    这个辅助函数封装了处理和保存单个文件的逻辑，并安全地更新计数器。
    """
    try:
        result = process_single_cif(args, cif_file)
        if result is not None:
            # 3. 使用正确的锁对象来保护计数器的“读-改-写”操作
            with lock:
                current_index = counter.value
                counter.value += 1
            save_single_sample(result, output_dir, split_name, current_index)
            return True # 表示成功
    except Exception as e:
        print(f"A task failed for {os.path.basename(cif_file)}: {e}")
    return False # 表示失败


def main(args):
    """Main function"""
    reset_seed()
    start_time = time.time()
    
    # Get list of CIF files
    cif_files = glob.glob(f"{args.cif_path}/*.cif")
    if args.debug > 0:
        print(f"Debug mode, randomly selecting {args.debug} CIF files")
        cif_files = random.sample(cif_files, min(args.debug, len(cif_files)))
    
    if not cif_files:
        print("No CIF files found. Exiting.")
        return
        
    print(f"Found {len(cif_files)} total CIF files.")
    
    # Create save directory and save config
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # Shuffle and split the list of CIF files BEFORE processing
    print("Shuffling and splitting CIF file list...")
    random.shuffle(cif_files)
    
    total_files = len(cif_files)
    test_size = int(total_files * args.test_ratio)
    val_size = int(total_files * args.val_ratio)
    
    test_cifs = cif_files[:test_size]
    val_cifs = cif_files[test_size : test_size + val_size]
    train_cifs = cif_files[test_size + val_size :]
    
    print(f"Dataset split (by files): Training {len(train_cifs)}, Val {len(val_cifs)}, Test {len(test_cifs)}")

    # Process each split
    train_count = process_and_save_split(args, train_cifs, 'train')
    val_count = process_and_save_split(args, val_cifs, 'val')
    test_count = process_and_save_split(args, test_cifs, 'test')
    
    total_samples_count = train_count + val_count + test_count
    
    # Final statistics
    print("\n" + "="*50)
    print("Dataset Creation Summary")
    print("="*50)
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
        "processing_time_seconds": time.time() - start_time
    }
    
    with open(os.path.join(args.save_path, "dataset_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print(f"\nDataset creation complete! Total time: {stats['processing_time_seconds']:.1f} seconds")
    print(f"Dataset saved to: {args.save_path}")
    print("Each sample is saved as an individual .jsonl file in its respective split folder.")


if __name__ == "__main__":
    cli_args = args()
    main(cli_args)
