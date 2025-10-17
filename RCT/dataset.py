import argparse
import json
import os
import random
import glob
import math

import torch
from torch.utils.data import IterableDataset, DataLoader

class MillerDataset(IterableDataset):
    def __init__(self, paths, miller_index_offset, augment_angle=False, augment_scale=False, 
                 scale_range=(0.9, 1.1), max_points=8192, debug=0, abs_label=False, norm_scale=False, fixed_clip_fraction=None, fixed_density_divisor=None,
                 lattice_stats_json=None, lattice_mean=None, lattice_std=None,
                 intensity_mean=None, intensity_std=None):
        super(MillerDataset, self).__init__()
        
        # Support single path or list of paths
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths
            
        self.files = []
        for path in self.paths:
            if not os.path.isdir(path):
                print(f"Warning: Not a valid directory, skipped: {path}")
                continue
                
            path_files = glob.glob(os.path.join(path, '**', '*.jsonl'), recursive=True)
            if not path_files:
                print(f"Warning: No '.jsonl' files found in directory {path}.")
            else:
                # print(f"--> Loaded {len(path_files)} files from {path}")
                self.files.extend(path_files)
        
        if debug > 0: 
            self.files = self.files[:debug]
            print(f"--> Debug mode: limited to first {debug} files")
            
        if not self.files:
            raise ValueError(f"No '.jsonl' files found in any provided path: {self.paths}")

        self.augment_angle = augment_angle
        self.augment_scale = augment_scale
        self.scale_range = scale_range
        self.max_points = max_points
        self.miller_index_offset = miller_index_offset
        self.abs_label = abs_label
        self.norm_scale = norm_scale
        self.fixed_clip_fraction = fixed_clip_fraction
        self.fixed_density_divisor = fixed_density_divisor

        # Intensity standardization params (if provided, use (I-mean)/std; else fallback to divide-by-10)
        self.intensity_mean = None if intensity_mean is None else float(intensity_mean)
        self.intensity_std = None if intensity_std is None else float(intensity_std)

        # Lattice normalization stats (mean/std per a,b,c,alpha,beta,gamma)
        self.lattice_mean = None
        self.lattice_std = None
        if lattice_mean is not None and lattice_std is not None:
            self.lattice_mean = torch.tensor(list(lattice_mean), dtype=torch.float)
            self.lattice_std = torch.tensor(list(lattice_std), dtype=torch.float)
        elif lattice_stats_json is not None and os.path.isfile(lattice_stats_json):
            try:
                with open(lattice_stats_json, 'r', encoding='utf-8') as f:
                    stats_payload = json.load(f)
                stats = stats_payload.get('stats', {})
                order = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
                means = [float(stats[k].get('mean', 0.0)) for k in order]
                stds = [float(stats[k].get('std', 1.0)) for k in order]
                # Avoid zero std
                stds = [s if abs(s) > 1e-8 else 1.0 for s in stds]
                self.lattice_mean = torch.tensor(means, dtype=torch.float)
                self.lattice_std = torch.tensor(stds, dtype=torch.float)
                print(f"--> Loaded lattice stats from {lattice_stats_json}")
            except Exception as e:
                print(f"Warning: failed to load lattice stats '{lattice_stats_json}': {e}")
                self.lattice_mean, self.lattice_std = None, None
        # Fallback: legacy coarse scaling if no stats provided
        if self.lattice_mean is None or self.lattice_std is None:
            self.lattice_mean = torch.zeros(6, dtype=torch.float)
            self.lattice_std = torch.tensor([10.0, 10.0, 10.0, 180.0, 180.0, 180.0], dtype=torch.float)
            print("--> Lattice stats not provided; using legacy coarse scaling [0 mean, std=(10,10,10,180,180,180)]")
        
        print(f"--> Total files loaded: {len(self.files)}")
        if self.norm_scale:
            print("--> Coordinate normalization enabled")
        else:
            print("--> Coordinate normalization disabled")

    def __len__(self):
        return len(self.files)

    def _apply_augmentations(self, points, labels):
        points_tensor = torch.tensor(points, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        original_mask = torch.ones(points_tensor.shape[0], dtype=torch.bool)

        if self.norm_scale:
            max_xy = torch.max(torch.abs(points_tensor[:, 1:3]))
            if max_xy > 0:
                scale_factor = 0.99 / max_xy
                points_tensor[:, 1:3] = (points_tensor[:, 1:3] - 0.5) * scale_factor + 0.5

        if self.augment_angle:
            shift = random.random()
            points_tensor[:, 0] = (points_tensor[:, 0] + shift) % 1.0
        
        # Deterministic angle clipping (for eval) has higher priority than random clipping
        if self.fixed_clip_fraction is not None:
            clip_fraction = float(self.fixed_clip_fraction)
            mask = points_tensor[:, 0] <= clip_fraction
            assert mask.sum() > 0
            points_tensor = points_tensor[mask]
            labels_tensor = labels_tensor[mask]

        if self.augment_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            points_tensor[:, 1:3] *= scale
            mask = (points_tensor[:, 1] >= 0) & (points_tensor[:, 1] <= 1) & (points_tensor[:, 2] >= 0) & (points_tensor[:, 2] <= 1)
            points_tensor = points_tensor[mask]
            if points_tensor.shape[0] == 0:
                return torch.tensor(points, dtype=torch.float), torch.ones(len(points), dtype=torch.bool)
            return points_tensor, mask, labels_tensor
        
        # Deterministic uniform angular downsampling by divisor (keep every k-th by sorted angle)
        if self.fixed_density_divisor is not None and int(self.fixed_density_divisor) > 1:
            k = int(self.fixed_density_divisor)
            order = torch.argsort(points_tensor[:, 0])
            kept = order[::k]
            if kept.numel() == 0:
                kept = order[:1]
            points_tensor = points_tensor[kept]
            labels_tensor = labels_tensor[kept]

        return points_tensor, original_mask, labels_tensor
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.files if worker_info is None else self.files[worker_info.id::worker_info.num_workers]
        random.shuffle(files_to_process)
        
        for f_path in files_to_process:
            filename = os.path.basename(f_path)
            with open(f_path, 'r') as f:
                line = f.readline()
                data = json.loads(line)
                assert len(data['input_sequence']) == len(data['labels']), f"input_sequence: {len(data['input_sequence'])}, labels: {len(data['labels'])}"
                # INSERT_YOUR_CODE
                if 'metadata' in data and 'crystal_params' in data['metadata']:
                    data['metadata']['crystal_params'].pop('_symmetry_space_group_name_H-M', None)
                assert len(data['metadata']['crystal_params'].values()) == 7, f"metadata: {data['metadata']}"

                points, aug_mask, labels_raw = self._apply_augmentations(data['input_sequence'], data['labels'])
                
                if labels_raw.ndim == 2:
                    # [N, 3] -> 单标签
                    labels = labels_raw
                else:
                    # [H, N, 3] -> 仅取第一个假设
                    labels = labels_raw[0]
                if self.augment_scale: labels = labels[aug_mask]
                num_points = points.shape[0]
                if num_points == 0: continue
                
                crystal_params_raw = data.get('metadata', {}).get('crystal_params', None)
                raw_vals = torch.tensor([float(p) for p in crystal_params_raw.values()][:6], dtype=torch.float)
                # Standardize using dataset stats
                lattice_labels = (raw_vals - self.lattice_mean) / self.lattice_std
                sg_label = torch.tensor([int(crystal_params_raw['_symmetry_Int_Tables_number']) - 1], dtype=torch.long)
                crystal_labels = {'lattice': lattice_labels, 'space_group': sg_label}

                assert torch.max(sg_label) <= 229 and torch.min(sg_label) >= 0
                
                # Intensity transform: standardize if mean/std provided; otherwise fallback to divide-by-10
                if (self.intensity_mean is not None) and (self.intensity_std is not None) and (abs(self.intensity_std) > 1e-12):
                    points[:, 3] = (points[:, 3] - float(self.intensity_mean)) / float(self.intensity_std)
                else:
                    points[:, 3] = points[:, 3] / 10.0
                if points.shape[0] > self.max_points:
                    points = points[:self.max_points]
                    labels = labels[:self.max_points, :]
                
                if self.abs_label:
                    labels = torch.abs(labels)
                else:
                    labels += self.miller_index_offset
                
                assert torch.max(labels) <= 10 and torch.min(labels) >= 0

                # Sample info
                sample_info = {
                    'filename': filename,
                    'source_path': f_path,
                }
                yield points, labels, crystal_labels, sample_info


def collate_fn_offset(batch):
    points_list, labels_list, crystal_labels_list, sample_info_list = zip(*batch)
    
    valid_indices = [i for i, p in enumerate(points_list) if p.shape[0] > 0]
    if not valid_indices:
        empty_crystal_labels = {'lattice': torch.tensor([]), 'space_group': torch.tensor([], dtype=torch.long)}
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int), empty_crystal_labels, []

    points_list = [points_list[i] for i in valid_indices]
    labels_list = [labels_list[i] for i in valid_indices]
    crystal_labels_list = [crystal_labels_list[i] for i in valid_indices]
    sample_info_list = [sample_info_list[i] for i in valid_indices]

    coords_list = [p[:, :3] for p in points_list]
    feats_list = [p[:, :] for p in points_list]
    
    coords = torch.cat(coords_list, dim=0).contiguous()
    feats = torch.cat(feats_list, dim=0).contiguous()
    labels = torch.cat(labels_list, dim=0).contiguous()
    
    offsets = torch.tensor([p.shape[0] for p in points_list], dtype=torch.int).cumsum(0)
    
    lattice_labels = torch.stack([c['lattice'] for c in crystal_labels_list], dim=0)
    sg_labels = torch.cat([c['space_group'] for c in crystal_labels_list], dim=0).to(torch.long)

    batched_crystal_labels = {'lattice': lattice_labels.contiguous(), 'space_group': sg_labels.contiguous()}
    
    return coords, feats, labels, offsets, batched_crystal_labels, sample_info_list
