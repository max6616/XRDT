from datad import *
import argparse
import numpy as np
from tqdm import tqdm
import os
import tifffile
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import glob
import json
import multiprocessing as mp
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
from PIL import Image
import random
from scipy.spatial.distance import pdist
from collections import defaultdict
import threading

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)


def reset_seed():
    # Use bitmask to ensure value range
    main_seed = (time.time_ns() + os.getpid() + threading.get_ident()) & 0xFFFFFFFF
    np.random.seed(main_seed)
    random.seed(main_seed + 1)

def random_init_equipment(args):
    reset_seed()

    s = args.scale
    ps = args.ps
    dist = random.randint(args.detector_dist[0], args.detector_dist[1])
    sizex = args.detector_size[0]
    sizey = args.detector_size[1]
    ponix = random.randint(args.detector_poni[0], args.detector_poni[1])
    poniy = random.randint(args.detector_poni[2], args.detector_poni[3])
    detector = Detector(normal=(0, 0, 1), vx=(1, 0, 0), 
                        sizex=sizex*ps*s, sizey=sizey*ps*s, dist=dist*s, 
                        ponix=ponix*ps*s, poniy=poniy*ps*s, ps=ps)
    # Apply small random perturbations to the detector in three directions
    angle_x = np.random.uniform(-1, 1)  # Randomly rotate ±0.5 degrees around x-axis
    angle_y = np.random.uniform(-1, 1)  # Randomly rotate ±0.5 degrees around y-axis
    angle_z = np.random.uniform(-1, 1)  # Randomly rotate ±0.5 degrees around z-axis
    
    detector.rotate_by_center(axis=(1, 0, 0), angle=angle_x, is_degree=True)
    detector.rotate_by_center(axis=(0, 1, 0), angle=angle_y, is_degree=True)
    detector.rotate_by_center(axis=(0, 0, 1), angle=angle_z, is_degree=True)
    
    
    mu = random.choice(args.xray_A)
    bdw = np.random.uniform(args.xray_bandwidth[0], args.xray_bandwidth[1])
    xray = Xray.Gaussian(mu=mu, sig=mu*bdw/2.355)

    eq_params = {'sizex': sizex, 
                 'sizey': sizey,
                 'ponix': ponix,
                 'poniy': poniy,
                 'mu': mu,
                 'bdw': bdw}

    return detector, xray, eq_params

def remove_low_ints(detector, meta_data, args):
    return [peak for peak in meta_data if detector.pic[peak['py'], peak['px']] > args.min_intensity]

def remove_overlap(meta_data):
    meta_data.sort(key=lambda x: x['intensity'], reverse=True)
    filtered_data = []
    used_peaks = {}  # Used to store used (hkl, position) combinations
    
    for peak in meta_data:
        px, py = peak['px'], peak['py']
        hkl = tuple(peak['hkl'])
        too_close = False
        
        # Check if there is already a peak with the same hkl too close
        for (used_hkl, used_x, used_y) in used_peaks:
            if hkl == used_hkl and ((px - used_x)**2 + (py - used_y)**2) < 4:
                too_close = True
                break
                
        if not too_close:
            filtered_data.append(peak)
            used_peaks[(hkl, px, py)] = peak['intensity']
            
    return filtered_data

def remove_invalid(args):
    """
    remove invalid images that have no peaks
    """
    train_files = glob.glob(os.path.join(args.save_path, "train2017", "*.png"))
    val_files = glob.glob(os.path.join(args.save_path, "val2017", "*.png"))
    
    removed = []
    for img_path in train_files + val_files:
        img = np.array(Image.open(img_path))
        if img.max() < 100:
            img_id = os.path.basename(img_path).split('.')[0]
            os.remove(img_path)
            removed.append(img_id)
            if img_path in train_files:
                ann_path = os.path.join(args.save_path, "annotations/train", f"{img_id}.json")
            else:
                ann_path = os.path.join(args.save_path, "annotations/val", f"{img_id}.json")
                
            if os.path.exists(ann_path):
                os.remove(ann_path)

def check_duplicate(meta_data, cif_file):
    """
    Check and report duplicate diffraction peaks with same Miller indices.
    """
    seen_hkls = {}
    hkl_positions = {}
    for peak in meta_data:
        hkl = tuple(peak['hkl'])
        if hkl not in seen_hkls:
            seen_hkls[hkl] = 1
            hkl_positions[hkl] = [(peak['px'], peak['py'])]
        else:
            seen_hkls[hkl] += 1
            hkl_positions[hkl].append((peak['px'], peak['py']))
    
    for hkl, count in seen_hkls.items():
        if count > 1:
            positions_str = ' '.join([f"({x}, {y})" for x, y in hkl_positions[hkl]])
            tqdm.write(f"{cif_file}: ({hkl[0]},{hkl[1]},{hkl[2]}) appears {count} times: {positions_str}")

def show_image(image, meta_data, args, file_name='test', aug=None):
    # Calculate the number of rows and columns for subplots
    try:
        num_frames = args.num_frames
    except:
        image = np.expand_dims(image, axis=-1)
        num_frames = 1
    n_cols = min(5, num_frames)  # Maximum 5 subplots per row
    n_rows = (num_frames + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create subplot layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    # Create visualization for each channel
    for channel_idx in range(num_frames):
        # Extract the image and metadata for the current channel
        channel_image = image[..., channel_idx]
        try: 
            if meta_data[0]['frame'] is not None:
                channel_meta = [peak for peak in meta_data if 'frame' in peak and peak['frame'] == channel_idx]
            else:
                channel_meta = meta_data
        except:
            return
        
        # Show image
        ax = axes[channel_idx]
        channel_image = np.sqrt(channel_image)
        im = ax.imshow(channel_image, cmap='gray')
        
        # Add annotations
        for peak in channel_meta:
            x, y, w, h = peak['bbox']
            hkl = peak['hkl']
            hkl_str = f"({hkl[0]},{hkl[1]},{hkl[2]})"
            label_str = f"{hkl_str}"
            
            ax.annotate(label_str,
                       xy=(x + w, y + h),
                       xytext=(1, 1),
                       textcoords='offset points',
                       fontsize=6,
                       color='yellow')
            rect = plt.Rectangle((x, y), w, h, color='yellow', fill=False, linewidth=0.4, alpha=0.5)
            ax.add_patch(rect)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide extra subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[-1], location='right')
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{args.save_path}/debug/vis/{file_name}.png", bbox_inches="tight", dpi=300)
    plt.close()

def get_min_bounding_box(bbox1, bbox2):
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2
    
    min_x1 = min(x1_1, x1_2)
    min_y1 = min(y1_1, y1_2)
    
    max_x2 = max(x1_1 + w1, x1_2 + w2)
    max_y2 = max(y1_1 + h1, y1_2 + h2)
    
    new_w = max_x2 - min_x1
    new_h = max_y2 - min_y1
    
    return [min_x1, min_y1, new_w, new_h]

def merge_peaks(meta_data):
    """
    Group peaks by hkl and frame, then merge overlapping ones, and return the processed list of diffraction points.
    Ensure that diffraction points from different frames are not merged incorrectly.
    """
    processed_meta_data = []
    
    # First group by hkl and frame
    groups = {}
    for peak in meta_data:
        key = (peak['hkl'], peak['frame'] if 'frame' in peak else None)
        if key not in groups:
            groups[key] = []
        groups[key].append(peak)
    
    # Merge peaks within each group
    for (hkl, frame), peaks in groups.items():
        if len(peaks) == 1:
            processed_meta_data.append({
                'hkl': hkl,
                'bbox': peaks[0]['bbox'],
                'frame': frame
            })
            continue
        
        # Merge peaks within the same group
        bboxes = [peak['bbox'] for peak in peaks]
        merged_bboxes = merge_overlapping_bboxes(bboxes)
        for b in merged_bboxes:
            processed_meta_data.append({
                'hkl': hkl,
                'bbox': b,
                'frame': frame
            })
    
    return processed_meta_data

def merge_overlapping_bboxes(bboxes):
    """
    Merge bboxes with transitive overlap and return the merged minimal bounding boxes.
    Args:
        bboxes: list of [x, y, w, h]
    Returns:
        list of merged [x, y, w, h]
    """
    n = len(bboxes)
    if n == 0:
        return []
    
    # Convert to corner representation (x1, y1, x2, y2)
    corners = []
    for bbox in bboxes:
        x, y, w, h = bbox
        corners.append( (x, y, x + w, y + h) )
    
    # Union-find initialization
    parent = list(range(n))
    rank = [1] * n
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        # Union by rank
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
    
    # Check all overlapping pairs
    for i in range(n):
        x1_i, y1_i, x2_i, y2_i = corners[i]
        for j in range(i + 1, n):
            x1_j, y1_j, x2_j, y2_j = corners[j]
            # Check if overlap
            overlap_x = x1_i <= x2_j and x2_i >= x1_j
            overlap_y = y1_i <= y2_j and y2_i >= y1_j
            if overlap_x and overlap_y:
                union(i, j)
    
    # Merge by group
    groups = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        groups[root].append(idx)
    
    merged = []
    for group in groups.values():
        min_x1 = min(corners[idx][0] for idx in group)
        min_y1 = min(corners[idx][1] for idx in group)
        max_x2 = max(corners[idx][2] for idx in group)
        max_y2 = max(corners[idx][3] for idx in group)
        new_w = max_x2 - min_x1
        new_h = max_y2 - min_y1
        merged.append( [min_x1, min_y1, new_w, new_h] )
    
    return merged

        
if __name__ == "__main__":
    image = tifffile.imread('datasets/test/train2017/3.tiff')
    show_image(image, )