import argparse
import os
import time
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from XRDT.model import XRDT
from XRDT.dataset import MillerDataset, collate_fn_offset
from XRDT.loss import CombinedLoss

warnings.filterwarnings("ignore")

from cctbx import crystal, miller
from cctbx.array_family import flex
from matplotlib.collections import PolyCollection

def get_parser():
    parser = argparse.ArgumentParser(description='High-Performance Point Transformer Evaluation')
    parser.add_argument('--model', type=str, default='pt_v3', help='Model architecture to use')
    parser.add_argument('--data_paths', nargs='+', default=[
        '/media/max/Data/datasets/mp_random_150k_v1_canonical', 
        '/media/max/Data/datasets/mp_random_150k_v2_canonical',
        '/media/max/Data/datasets/mp_random_150k_v3_canonical'
        ], help='Multiple dataset paths, separated by spaces')
    parser.add_argument('--save_path', type=str, default='./eval_results', help='Path to save evaluation results')
    parser.add_argument('--checkpoint', type=str, default='/media/max/Data/results/xrd_transformer/v1-3_canonical_ang_clip_density_clip_noise/XRDT_20251002_170401/best_model.pth', help='Model checkpoint path to evaluate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker processes for data loading')
    parser.add_argument('--min_hkl', type=int, default=-5, help='Global minimum value of hkl in dataset (obtained from analysis script)')
    parser.add_argument('--max_hkl', type=int, default=5, help='Global maximum value of hkl in dataset (obtained from analysis script)')
    # Single label evaluation mode, no multi-hypothesis parameters needed
    parser.add_argument('--abs_label', type=bool, default=False, help='Whether to use absolute values')
    parser.add_argument('--norm_scale', default=True, help='Enable coordinate scaling normalization')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode, 0 means disabled')
    parser.add_argument('--eval_set', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset to evaluate')
    # Angle range evaluation related
    parser.add_argument('--angle_eval_ranges', nargs='+', type=float, default=[1.0, 0.5, 0.25, 0.16667, 0.08334, 0.02778], help='Proportional angle clipping ranges (1.0=360°, 0.5=180°, 0.1667≈60°, 0.0278≈10°)')
    # HKL canonicalization control
    parser.add_argument('--canonicalize_hkl', type=str, default='none', choices=['none', 'gt', 'pred'], help='Whether to perform ASU canonicalization on predicted HKL, using gt or pred space group')
    # Density sensitivity evaluation on full-angle data
    parser.add_argument('--density_eval_ranges', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], help='Uniform angular downsampling divisors, e.g., 2 4 8 -> 1/2, 1/4, 1/8')
    # Noisy evaluation (training-like perturbations during eval)
    parser.add_argument('--noisy_eval_enable', type=bool, default=True, help='Run an extra evaluation with noise perturbation')
    parser.add_argument('--noise_eval_prob', type=float, default=0.05, help='Probability of turning a point into full random noise (ignored in loss/metrics)')
    parser.add_argument('--noise_eval_micro', type=float, default=0.01, help='Max micro perturbation on XY (in [0,1] range)')
    return parser

def evaluate(loader, model, criterion, save_path=None, eval_set_name='val', canonicalize_hkl='none', miller_index_offset=0, apply_noise=False, noise_prob=0.0, noise_micro=0.01):
    model.eval()
    total_loss, h_correct, k_correct, l_correct, all_correct, total_points = 0, 0, 0, 0, 0, 0
    hkl_loss, lattice_loss, sg_loss = 0, 0, 0
    total_lattice_mae_a, total_lattice_mae_ang = 0, 0
    sg_correct_total, total_samples_with_crystal_info = 0, 0
    
    # For collecting detailed information of each sample
    sample_details = []
    
    # For collecting lattice parameter error data
    abc_errors_all = []
    ang_errors_all = []
    
    # Initialize extended statistics dictionary to store all required metrics
    stats_by_sg = {i: {
        'sg_correct': 0, 'sg_total': 0, 'h_correct': 0, 'k_correct': 0, 
        'l_correct': 0, 'all_correct': 0, 'total_points': 0
    } for i in range(230)}

    crystal_systems = {
        'triclinic': list(range(1, 3)), 'monoclinic': list(range(3, 16)),
        'orthorhombic': list(range(16, 75)), 'tetragonal': list(range(75, 143)),
        'trigonal': list(range(143, 168)), 'hexagonal': list(range(168, 195)),
        'cubic': list(range(195, 231))
    }

    # Per-space-group accuracy distributions by sample for violin plots
    accs_h_by_sg = {i: [] for i in range(230)}
    accs_k_by_sg = {i: [] for i in range(230)}
    accs_l_by_sg = {i: [] for i in range(230)}
    accs_all_by_sg = {i: [] for i in range(230)}

    def _canonicalize_hkl_batch_cctbx(hkl_triplets, sg_number_1_based):
        """
        hkl_triplets: List[Tuple[int,int,int]]
        sg_number_1_based: int in [1,230]
        returns List[List[int]] same length
        """

        symm = crystal.symmetry(space_group_symbol=str(int(sg_number_1_based)))
        ms = miller.set(
            crystal_symmetry=symm,
            indices=flex.miller_index(hkl_triplets),
            anomalous_flag=False
        )
        ms_asu = ms.map_to_asu()
        idx = ms_asu.indices()
        return [list(idx[i]) for i in range(len(hkl_triplets))]
    
    def _apply_noise_adding_batch(coords, feats, offsets, micro_noise_max: float = 0.01, prob_noisify: float = 0.0):
        # Keep angle unchanged; add small XY noise; optionally fully randomize XY for a subset
        if micro_noise_max > 0.0:
            n_points = feats.shape[0]
            xy_noise = (torch.rand((n_points, 2), device=feats.device) * 2.0 - 1.0) * float(micro_noise_max)
            coords[:, 1:3] = torch.clamp(coords[:, 1:3] + xy_noise, 0.0, 1.0)
            feats[:, 1:3] = torch.clamp(feats[:, 1:3] + xy_noise, 0.0, 1.0)
        loss_mask = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
        if prob_noisify > 0.0:
            n_points = feats.shape[0]
            full_noise_mask = (torch.rand((n_points,), device=feats.device) < float(prob_noisify))
            if torch.any(full_noise_mask):
                rand_xy = torch.rand((int(full_noise_mask.sum().item()), 2), device=feats.device)
                coords[full_noise_mask, 1:3] = rand_xy
                feats[full_noise_mask, 1:3] = rand_xy
                loss_mask[full_noise_mask] = False
        return coords, feats, loss_mask

    with torch.no_grad():
        try:
            total_samples_for_pbar = len(loader.dataset)
        except Exception:
            total_samples_for_pbar = None

        pbar = tqdm(total=total_samples_for_pbar, desc='Evaluating', unit='samples', dynamic_ncols=True)
        for i, (coords, feats, miller_labels, offsets, crystal_labels, sample_info_list) in enumerate(loader):
            coords, feats, miller_labels, offsets = coords.cuda(non_blocking=True), feats.cuda(non_blocking=True), miller_labels.cuda(non_blocking=True), offsets.cuda(non_blocking=True)
            crystal_labels = {k: v.cuda(non_blocking=True) for k, v in crystal_labels.items()}

            point_loss_mask = None
            if apply_noise:
                coords, feats, point_loss_mask = _apply_noise_adding_batch(coords, feats, offsets, micro_noise_max=noise_micro, prob_noisify=noise_prob)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(coords, feats, offsets)
                loss_dict = criterion(predictions, miller_labels, crystal_labels, offsets, point_loss_mask=point_loss_mask)
            
            total_loss += loss_dict['total_loss'].detach().item() * len(offsets)
            hkl_loss += loss_dict['loss_miller'].detach().item() * len(offsets)
            lattice_loss += loss_dict['loss_lattice'].detach().item() * len(offsets)
            sg_loss += loss_dict['loss_sg'].detach().item() * len(offsets)

            # --- Collect hkl accuracy by sample (single label) ---
            pred_h_all = torch.argmax(predictions['h'], dim=1)
            pred_k_all = torch.argmax(predictions['k'], dim=1)
            pred_l_all = torch.argmax(predictions['l'], dim=1)
            if apply_noise and point_loss_mask is not None:
                mask_all = point_loss_mask
            else:
                mask_all = torch.ones_like(pred_h_all, dtype=torch.bool, device=pred_h_all.device)
            
            start_idx = 0
            for sample_idx in range(len(offsets)):
                end_idx = offsets[sample_idx]
                sample_mask = mask_all[start_idx:end_idx]
                num_points_in_sample = int(sample_mask.sum().item())
                
                # Directly use single label [N,3]
                sample_labels_flat = miller_labels[start_idx:end_idx]
                labels_h_all = sample_labels_flat[:, 0]
                labels_k_all = sample_labels_flat[:, 1]
                labels_l_all = sample_labels_flat[:, 2]
                
                # Get predictions for this sample
                pred_h = pred_h_all[start_idx:end_idx]
                pred_k = pred_k_all[start_idx:end_idx]
                pred_l = pred_l_all[start_idx:end_idx]

                # Optional: perform ASU canonicalization on predicted HKL
                if canonicalize_hkl != 'none':
                    if canonicalize_hkl == 'gt':
                        sg = crystal_labels['space_group'][sample_idx].item() + 1
                    else:  # 'pred'
                        sg = int(torch.argmax(predictions['space_group'][sample_idx]).item()) + 1

                    # Decode to integer HKL
                    h_int = (pred_h - miller_index_offset).detach().cpu().numpy().astype(int)
                    k_int = (pred_k - miller_index_offset).detach().cpu().numpy().astype(int)
                    l_int = (pred_l - miller_index_offset).detach().cpu().numpy().astype(int)
                    triplets = list(zip(h_int.tolist(), k_int.tolist(), l_int.tolist()))
                    # print(sg)

                    canon_triplets = _canonicalize_hkl_batch_cctbx(triplets, sg)
                    # Re-encode to category indices
                    canon_h = torch.tensor([t[0] + miller_index_offset for t in canon_triplets], device=pred_h.device, dtype=pred_h.dtype)
                    canon_k = torch.tensor([t[1] + miller_index_offset for t in canon_triplets], device=pred_k.device, dtype=pred_k.dtype)
                    canon_l = torch.tensor([t[2] + miller_index_offset for t in canon_triplets], device=pred_l.device, dtype=pred_l.dtype)

                    pred_h, pred_k, pred_l = canon_h, canon_k, canon_l
                
                # Calculate correct counts for this sample (masked if noisy)
                if num_points_in_sample > 0:
                    labels_h = labels_h_all[sample_mask]
                    labels_k = labels_k_all[sample_mask]
                    labels_l = labels_l_all[sample_mask]
                    pred_h_eff = pred_h[sample_mask]
                    pred_k_eff = pred_k[sample_mask]
                    pred_l_eff = pred_l[sample_mask]
                    sample_h_correct = (pred_h_eff == labels_h).sum().item()
                    sample_k_correct = (pred_k_eff == labels_k).sum().item()
                    sample_l_correct = (pred_l_eff == labels_l).sum().item()
                    sample_all_correct = ((pred_h_eff == labels_h) & (pred_k_eff == labels_k) & (pred_l_eff == labels_l)).sum().item()
                else:
                    sample_h_correct = 0
                    sample_k_correct = 0
                    sample_l_correct = 0
                    sample_all_correct = 0
                
                # Calculate accuracy for this sample
                sample_accuracy = sample_all_correct / num_points_in_sample * 100 if num_points_in_sample > 0 else 0
                
                # Get sample information
                sample_info = sample_info_list[sample_idx]
                sg_idx = crystal_labels['space_group'][sample_idx].item()

                # Per-sample h/k/l accuracies
                sample_acc_h = sample_h_correct / num_points_in_sample * 100 if num_points_in_sample > 0 else 0
                sample_acc_k = sample_k_correct / num_points_in_sample * 100 if num_points_in_sample > 0 else 0
                sample_acc_l = sample_l_correct / num_points_in_sample * 100 if num_points_in_sample > 0 else 0
                
                # Collect sample detailed information
                sample_detail = {
                    'filename': sample_info['filename'],
                    'accuracy': float(sample_accuracy),
                    'space_group': int(sg_idx),
                    'total_points': int(num_points_in_sample),
                    'correct_points': int(sample_all_correct)
                }
                # Add predicted SG (0-based) if available
                try:
                    pred_sg_idx = int(torch.argmax(predictions['space_group'][sample_idx]).item())
                except Exception:
                    pred_sg_idx = -1
                sample_detail['pred_sg'] = pred_sg_idx
                sample_details.append(sample_detail)
                
                # Add to global statistics
                h_correct += sample_h_correct
                k_correct += sample_k_correct
                l_correct += sample_l_correct
                all_correct += sample_all_correct
                total_points += num_points_in_sample
                
                # If space group information exists, add to space group-wise statistics
                if sg_idx != -1:
                    stats_by_sg[sg_idx]['h_correct'] += sample_h_correct
                    stats_by_sg[sg_idx]['k_correct'] += sample_k_correct
                    stats_by_sg[sg_idx]['l_correct'] += sample_l_correct
                    stats_by_sg[sg_idx]['all_correct'] += sample_all_correct
                    stats_by_sg[sg_idx]['total_points'] += num_points_in_sample
                    # Also collect per-sample accuracy distributions for violin plots
                    accs_h_by_sg[sg_idx].append(float(sample_acc_h))
                    accs_k_by_sg[sg_idx].append(float(sample_acc_k))
                    accs_l_by_sg[sg_idx].append(float(sample_acc_l))
                    accs_all_by_sg[sg_idx].append(float(sample_accuracy))
                
                start_idx = end_idx
                
            # --- Collect lattice and space group accuracy ---
            valid_crystal_mask = (crystal_labels['space_group'] != -1).squeeze()
            num_valid_samples = valid_crystal_mask.sum().item()
            if num_valid_samples > 0:
                pred_lattice = predictions['lattice_params'][valid_crystal_mask]
                target_lattice = crystal_labels['lattice'][valid_crystal_mask]
                pred_lattice_unnorm = pred_lattice.clone(); target_lattice_unnorm = target_lattice.clone()
                pred_lattice_unnorm[:, :3] *= 10; target_lattice_unnorm[:, :3] *= 10
                pred_lattice_unnorm[:, 3:] *= 180; target_lattice_unnorm[:, 3:] *= 180
                
                # Calculate absolute errors for abc and angles
                abc_errors = torch.abs(pred_lattice_unnorm[:, :3] - target_lattice_unnorm[:, :3])
                ang_errors = torch.abs(pred_lattice_unnorm[:, 3:] - target_lattice_unnorm[:, 3:])
                
                # Accumulate MAE
                total_lattice_mae_a += torch.nn.functional.l1_loss(pred_lattice_unnorm[:, :3], target_lattice_unnorm[:, :3], reduction='sum').item()
                total_lattice_mae_ang += torch.nn.functional.l1_loss(pred_lattice_unnorm[:, 3:], target_lattice_unnorm[:, 3:], reduction='sum').item()
                
                # Collect all errors for cumulative distribution plots
                abc_errors_all.extend(abc_errors.flatten().cpu().numpy())
                ang_errors_all.extend(ang_errors.flatten().cpu().numpy())

                pred_sg = torch.argmax(predictions['space_group'][valid_crystal_mask], dim=1)
                target_sg = crystal_labels['space_group'][valid_crystal_mask].squeeze()
                sg_correct_total += (pred_sg == target_sg).sum().item()
                total_samples_with_crystal_info += num_valid_samples

            # Update progress bar by number of samples in this batch
            try:
                batch_num_samples = int(offsets.numel())
            except Exception:
                batch_num_samples = 1
            pbar.update(batch_num_samples)

        pbar.close()

    # --- Plotting section ---
    if total_samples_with_crystal_info > 0 and save_path is not None:
        print(f"[{eval_set_name.upper()}] Generating and saving evaluation charts...")
        # Generate median±IQR summary scatter plots colored by crystal systems
        plot_summary_accuracy_by_sg(accs_h_by_sg, crystal_systems, save_path, eval_set_name, 'h')
        plot_summary_accuracy_by_sg(accs_k_by_sg, crystal_systems, save_path, eval_set_name, 'k')
        plot_summary_accuracy_by_sg(accs_l_by_sg, crystal_systems, save_path, eval_set_name, 'l')
        plot_summary_accuracy_by_sg(accs_all_by_sg, crystal_systems, save_path, eval_set_name, 'all')
        # Additionally, violin plot aggregated by crystal systems (overall accuracy)
        plot_violin_accuracy_all_by_cs(accs_all_by_sg, crystal_systems, save_path, eval_set_name)
        
        # Generate lattice parameter cumulative distribution plots
        if len(abc_errors_all) > 0:
            plot_lattice_cumulative_distribution(abc_errors_all, ang_errors_all, save_path, eval_set_name)

    # Save sample detailed information to JSON file
    if save_path is not None and sample_details:
        # Sort by accuracy from low to high
        sample_details_sorted = sorted(sample_details, key=lambda x: x['accuracy'])

        # Ensure eval_results directory exists
        eval_results_path = save_path
        os.makedirs(eval_results_path, exist_ok=True)
        
        # Save JSON file
        json_filename = f'{eval_set_name}_sample_details.json'
        json_path = os.path.join(eval_results_path, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_details_sorted, f, indent=2, ensure_ascii=False)
        
        print(f"[{eval_set_name.upper()}] Sample details saved to: {json_path}")
        print(f"[{eval_set_name.upper()}] Total samples: {len(sample_details_sorted)}")
        if len(sample_details_sorted) > 0:
            print(f"[{eval_set_name.upper()}] Accuracy range: {sample_details_sorted[0]['accuracy']:.2f}% - {sample_details_sorted[-1]['accuracy']:.2f}%")

    num_total_samples = len(loader.dataset)
    metrics = {
        'loss': total_loss / num_total_samples if num_total_samples > 0 else 0,
        'loss_miller': hkl_loss / num_total_samples if num_total_samples > 0 else 0,
        'loss_lattice': lattice_loss / num_total_samples if num_total_samples > 0 else 0,
        'loss_sg': sg_loss / num_total_samples if num_total_samples > 0 else 0,
        'acc_h': h_correct / total_points * 100 if total_points > 0 else 0,
        'acc_k': k_correct / total_points * 100 if total_points > 0 else 0,
        'acc_l': l_correct / total_points * 100 if total_points > 0 else 0,
        'acc_all': all_correct / total_points * 100 if total_points > 0 else 0,
        'lattice_mae_a': total_lattice_mae_a / total_samples_with_crystal_info if total_samples_with_crystal_info > 0 else 0,
        'lattice_mae_ang': total_lattice_mae_ang / total_samples_with_crystal_info if total_samples_with_crystal_info > 0 else 0,
        'sg_acc': sg_correct_total / total_samples_with_crystal_info * 100 if total_samples_with_crystal_info > 0 else 0,
    }
    extras = {
        'abc_errors': abc_errors_all,
        'ang_errors': ang_errors_all,
        'stats_by_sg': stats_by_sg
    }
    return metrics, extras

def plot_accuracy_by_sg(stats_by_sg, crystal_systems, save_path, eval_set_name, acc_type):
    """
    Plot accuracy scatter plots by space group based on provided statistics.

    Args:
        stats_by_sg (dict): Dictionary containing space group-wise statistics.
        crystal_systems (dict): Mapping from crystal systems to space group number ranges.
        save_path (str): Root directory to save plots.
        eval_set_name (str): Name of evaluation set ('train' or 'val').
        acc_type (str): Type of accuracy to plot ('h', 'k', 'l', 'all').
    """
    accuracies = []
    sizes = []
    systems = []

    for sg_idx in range(230):
        stats = stats_by_sg.get(sg_idx, {})
        total_points = stats.get('total_points', 0)
        
        if acc_type == 'all':
            correct_points = stats.get('all_correct', 0)
            title = 'Overall HKL Accuracy'
        else: # h, k, or l
            correct_points = stats.get(f'{acc_type}_correct', 0)
            title = f'{acc_type.upper()} Accuracy'

        # Ensure numeric conversion to Python types to prevent tensor-caused plotting errors
        if hasattr(correct_points, 'item'):
            correct_points = correct_points.item()
        if hasattr(total_points, 'item'):
            total_points = total_points.item()
            
        acc = (correct_points / total_points * 100) if total_points > 0 else 0
        accuracies.append(acc)
        
        # Set point size based on data point count, minimum 10, maximum 200
        point_size = max(10, min(200, total_points / 10)) if total_points > 0 else 20
        sizes.append(point_size)
        
        system_label = None
        for system, sg_range in crystal_systems.items():
            if sg_idx + 1 in sg_range:
                system_label = system
                break
        systems.append(system_label if system_label is not None else 'unknown')
    
    plt.figure(figsize=(22, 8))
    x_positions = list(range(1, 231))
    import pandas as pd
    df = pd.DataFrame({'sg': x_positions, 'acc': accuracies, 'size': sizes, 'system': systems})
    sns.set_theme(style='whitegrid')
    sns.scatterplot(data=df, x='sg', y='acc', hue='system', palette=sns.color_palette(), size='size', sizes=(10, 200), edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # plt.xlabel('Space Group Number', fontsize=12)
    # plt.ylabel('Accuracy (%)', fontsize=12)
    # plt.title(f'{title} by Space Group ({eval_set_name.capitalize()} Set)', fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(-2, 233)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.legend(loc='upper right', title='Crystal Systems')
    
    # Set x-axis ticks
    plt.xticks(range(0, 231, 20))
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save plot
    filename = f'{eval_set_name}_accuracy_{acc_type}_by_sg.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lattice_cumulative_distribution(abc_errors, ang_errors, save_path, eval_set_name):
    """
    Plot cumulative distribution charts for lattice parameters.
    
    Args:
        abc_errors (list): List of absolute errors for abc parameters
        ang_errors (list): List of absolute errors for angle parameters
        save_path (str): Directory to save plots
        eval_set_name (str): Name of evaluation set
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot cumulative distribution for abc parameters
    if len(abc_errors) > 0:
        abc_errors = np.array(abc_errors)
        # Define error range: 0.01 to 1
        abc_thresholds = np.logspace(-2, 0, 1000)  # 0.01 to 1, log distribution
        
        abc_cumulative = []
        for threshold in abc_thresholds:
            cumulative_ratio = np.sum(abc_errors <= threshold) / len(abc_errors) * 100
            abc_cumulative.append(cumulative_ratio)
        
        sns.lineplot(x=abc_thresholds, y=abc_cumulative, ax=ax1, linewidth=2, color=sns.color_palette()[0])
        ax1.set_xscale('log')
        ax1.set_xlabel('Absolute Error in ABC (Å)', fontsize=12)
        ax1.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax1.set_title(f'Cell Length Accuracy Distribution\n({eval_set_name.capitalize()} Set)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.01, 1.0)
        ax1.set_ylim(0, 100)
        
        # Add annotations for key points
        for threshold in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if threshold in abc_thresholds:
                idx = np.argmin(np.abs(abc_thresholds - threshold))
                ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
                ax1.text(threshold*1.2, abc_cumulative[idx], f'{abc_cumulative[idx]:.1f}%', 
                        fontsize=10, ha='left', va='center')
    
    # Plot cumulative distribution for angle parameters
    if len(ang_errors) > 0:
        ang_errors = np.array(ang_errors)
        # Define error range: 0.1 to 10 degrees
        ang_thresholds = np.logspace(-1, 1, 1000)  # 0.1 to 10, log distribution
        
        ang_cumulative = []
        for threshold in ang_thresholds:
            cumulative_ratio = np.sum(ang_errors <= threshold) / len(ang_errors) * 100
            ang_cumulative.append(cumulative_ratio)
        
        sns.lineplot(x=ang_thresholds, y=ang_cumulative, ax=ax2, linewidth=2, color=sns.color_palette()[1])
        ax2.set_xscale('log')
        ax2.set_xlabel('Absolute Error in Angles (degrees)', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title(f'Cell Angles Accuracy Distribution\n({eval_set_name.capitalize()} Set)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.1, 10.0)
        ax2.set_ylim(0, 100)
        
        # Add annotations for key points
        for threshold in [0.1, 0.5, 1.0, 5.0, 10.0]:
            if threshold in ang_thresholds:
                idx = np.argmin(np.abs(ang_thresholds - threshold))
                ax2.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
                ax2.text(threshold*1.2, ang_cumulative[idx], f'{ang_cumulative[idx]:.1f}%', 
                        fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    
    # Ensure save directory exists
    eval_results_path = save_path
    os.makedirs(eval_results_path, exist_ok=True)
    
    # Save figure
    filename = f'{eval_set_name}_lattice_cumulative_distribution.png'
    plt.savefig(os.path.join(eval_results_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{eval_set_name.upper()}] Lattice cumulative distribution charts saved to: {filename}")

def plot_violin_accuracy_by_sg(accs_by_sg, crystal_systems, save_path, eval_set_name, acc_type):
    """
    Draw per-space-group accuracy distribution violin plot using seaborn.

    Args:
        accs_by_sg (dict[int, list[float]]): Mapping from 0-based sg index to list of accuracies per sample.
        crystal_systems (dict[str, list[int]]): 1-based sg numbers categorized by crystal systems.
        save_path (str): Directory to save the plot.
        eval_set_name (str): Name of evaluation set.
        acc_type (str): One of 'h', 'k', 'l', 'all'.
    """
    # Build records for DataFrame
    records = []
    for sg_idx_0 in range(230):
        sg_num_1 = sg_idx_0 + 1
        acc_list = accs_by_sg.get(sg_idx_0, [])
        if not acc_list:
            continue
        # Locate crystal system label
        system_label = None
        for system_name, sg_range in crystal_systems.items():
            if sg_num_1 in sg_range:
                system_label = system_name
                break
        for acc in acc_list:
            records.append({
                'space_group': sg_num_1,
                'accuracy': float(acc),
                'system': system_label if system_label is not None else 'unknown'
            })

    if len(records) == 0:
        return

    df = pd.DataFrame.from_records(records)

    # Configure seaborn aesthetics
    sns.set_theme(style='whitegrid')

    # Determine title
    if acc_type == 'all':
        title = 'Overall HKL Accuracy'
    else:
        title = f'{acc_type.upper()} Accuracy'

    # Order x by increasing space group number
    order = sorted(df['space_group'].unique())

    # Create large figure due to many categories
    plt.figure(figsize=(24, 8))
    ax = sns.violinplot(
        data=df,
        x='space_group',
        y='accuracy',
        order=order,
        cut=0,
        scale='width',
        inner='quartile',
        linewidth=0.6,
        palette=sns.color_palette()
    )

    # ax.set_xlabel('Space Group Number', fontsize=12)
    # ax.set_ylabel('Accuracy (%)', fontsize=12)
    # ax.set_title(f'{title} by Space Group ({eval_set_name.capitalize()} Set)', fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xlim(-2, len(order) + 2)

    # Reduce xtick density: show every 20 if too many
    try:
        xticks = ax.get_xticks()
        if len(xticks) > 0:
            # Map tick indices to actual sg numbers in 'order'
            tick_positions = []
            tick_labels = []
            for idx, sg in enumerate(order):
                if sg % 20 == 0 or sg in (1, 230):
                    tick_positions.append(idx)
                    tick_labels.append(str(sg))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0)
    except Exception:
        pass

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f'{eval_set_name}_accuracy_{acc_type}_by_sg.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_accuracy_by_sg(accs_by_sg, crystal_systems, save_path, eval_set_name, acc_type):
    """
    Draw per-space-group accuracy summary as median with IQR, colored by crystal system.
    """
    palette = sns.color_palette()
    system_names = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
    system_to_color = {name: palette[i % len(palette)] for i, name in enumerate(system_names)}

    x_positions = []
    medians = []
    iqr_lows = []
    iqr_highs = []
    sizes = []
    colors = []

    for sg_idx_0 in range(230):
        acc_list = accs_by_sg.get(sg_idx_0, [])
        if not acc_list:
            continue
        arr = np.asarray(acc_list, dtype=np.float32)
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        count = int(arr.size)

        sg_num_1 = sg_idx_0 + 1
        x_positions.append(sg_num_1)
        medians.append(median)
        iqr_lows.append(median - q1)
        iqr_highs.append(q3 - median)
        sizes.append(max(5, min(300, count)))

        system_label = None
        for system_name, sg_range in crystal_systems.items():
            if sg_num_1 in sg_range:
                system_label = system_name
                break
        colors.append(system_to_color.get(system_label, palette[0]))

    if len(x_positions) == 0:
        return

    plt.figure(figsize=(22, int(4)))
    # Draw IQR as vertical error bars (thinner to keep detail)
    for x, y, lo, hi, c in zip(x_positions, medians, iqr_lows, iqr_highs, colors):
        sns.lineplot(x=[x, x], y=[y - lo, y + hi], color=c, linewidth=0.6)
    # Draw medians as scatter, size by sample count
    import pandas as pd
    df = pd.DataFrame({
        'sg': x_positions,
        'median': medians,
        'size': sizes,
        'color': colors
    })
    # seaborn doesn't support per-point color array directly with hue legend here; use matplotlib scatter but with seaborn palette colors
    plt.scatter(df['sg'], df['median'], c=df['color'], s=df['size'], alpha=0.85, edgecolors='black', linewidths=0.3)

    if acc_type == 'all':
        title = 'Overall HKL Accuracy'
    else:
        title = f'{acc_type.upper()} Accuracy'

    # plt.xlabel('Space Group Number', fontsize=12)
    # plt.ylabel('Accuracy (%)', fontsize=12)
    # plt.title(f'{title} by Space Group (Median ± IQR, {eval_set_name.capitalize()} Set)', fontsize=14)
    plt.ylim(0, 105)
    plt.xlim(0, 231)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Legend for crystal systems (use circles instead of rectangles)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=system_to_color[name], 
               markeredgecolor='black', label=name, alpha=0.6, markersize=10)
        for name in system_names
    ]
    plt.legend(handles=legend_elements, loc='lower right', title='Crystal Systems')

    # Sparse xticks
    plt.xticks(range(0, 231, 10))

    os.makedirs(save_path, exist_ok=True)
    filename = f'{eval_set_name}_accuracy_{acc_type}_by_sg.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_violin_accuracy_all_by_cs(accs_all_by_sg, crystal_systems, save_path, eval_set_name):
    """
    Plot accuracy distribution grouped by crystal systems (violin plot) using seaborn.
    Saves as {eval_set_name}_accuracy_all_by_cs.png
    """
    # Aggregate per-space-group accuracies into crystal systems
    system_to_accs = {name: [] for name in crystal_systems.keys()}
    for system_name, sg_range in crystal_systems.items():
        for sg_num_1 in sg_range:
            acc_list = accs_all_by_sg.get(sg_num_1 - 1, [])
            if acc_list:
                system_to_accs[system_name].extend([float(a) for a in acc_list])

    # Build dataframe
    import pandas as pd
    records = []
    for system_name, accs in system_to_accs.items():
        for a in accs:
            records.append({'system': system_name, 'accuracy': a})

    if len(records) == 0:
        return

    df = pd.DataFrame.from_records(records)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(16, 6))
    order = list(crystal_systems.keys())
    ax = sns.violinplot(
        data=df,
        x='system',
        y='accuracy',
        order=order,
        cut=0,
        scale='width',
        inner='quartile',
        linewidth=0.8,
        palette=sns.color_palette(),
        bw_adjust=0.4,
        gridsize=256,
        saturation=0.9
    )
    # Increase spacing between groups
    ax.set_xlim(-0.6, len(order) - 0.4)
    # ax.set_xlabel('Crystal System', fontsize=12)
    # ax.set_ylabel('Accuracy (%)', fontsize=12)
    # ax.set_title(f'Overall HKL Accuracy by Crystal System ({eval_set_name.capitalize()} Set)', fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.8, len(order) - 0.2)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    filename = f'{eval_set_name}_accuracy_all_by_cs.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_violin_accuracy_all_by_cs_across_angles(root_save_dir, angle_ranges, angle_name_map, eval_set_name, crystal_systems):
    """
    Create a 3D figure where each y-slice represents an angle range, and for each slice,
    a set of violin-like surfaces along x (crystal systems) shows the distribution of overall accuracy.
    Data are read from {subdir}/{eval_set_name}_sample_details.json produced during evaluation.
    """
    import json as _json
    systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
    palette = sns.color_palette()
    sys_to_color = {name: palette[i % len(palette)] for i, name in enumerate(systems)}

    # Collect per-angle, per-system accuracies
    data_by_angle = []
    y_labels = []
    for frac in angle_ranges:
        name = angle_name_map[frac]
        subdir = os.path.join(root_save_dir, name)
        details_path = os.path.join(subdir, f'{eval_set_name}_sample_details.json')
        sys_accs = {s: [] for s in systems}
        if os.path.exists(details_path):
            with open(details_path, 'r', encoding='utf-8') as f:
                arr = _json.load(f)
            for rec in arr:
                sg_num_1 = int(rec.get('space_group', -1)) + 1 if int(rec.get('space_group', -1)) < 1 else int(rec.get('space_group', -1))
                # Accept both 0-based and 1-based inputs
                if sg_num_1 < 1 or sg_num_1 > 230:
                    continue
                system_name = None
                for k, rng in crystal_systems.items():
                    if sg_num_1 in rng:
                        system_name = k
                        break
                if system_name is None:
                    continue
                acc = float(rec.get('accuracy', 0.0))
                if np.isfinite(acc):
                    sys_accs[system_name].append(acc)
        data_by_angle.append(sys_accs)
        y_labels.append(name)

    if len(data_by_angle) == 0:
        return

    # Build 3D plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    z_grid = np.linspace(0.0, 100.0, 200)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    max_half_width = 0.35

    for y_idx, sys_accs in enumerate(data_by_angle):
        for x_idx, system_name in enumerate(systems):
            accs = sys_accs.get(system_name, [])
            if accs is None or len(accs) == 0:
                continue
            hist, _ = np.histogram(accs, bins=len(z_grid), range=(0.0, 100.0), density=True)
            density = np.convolve(hist, kernel, mode='same')
            density = density / (density.max() + 1e-8)
            half = density * max_half_width

            x_center = float(x_idx)
            x_left = x_center - half
            x_right = x_center + half
            xs = np.concatenate([x_left, x_right[::-1]])
            zs = np.concatenate([z_grid, z_grid[::-1]])
            verts = [list(zip(xs, zs))]
            color = sys_to_color[system_name]
            poly = PolyCollection(verts, facecolors=[color], edgecolors='black', linewidths=0.2, alpha=0.7)
            ax.add_collection3d(poly, zs=y_idx, zdir='y')

    ax.set_xlim(-0.5, len(systems) - 0.5)
    ax.set_ylim(-0.5, len(data_by_angle) - 0.5)
    ax.set_zlim(0.0, 100.0)
    ax.set_xticks(list(range(len(systems))))
    ax.set_xticklabels(systems, rotation=0)
    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(y_labels)
    ax.set_zlabel('Accuracy (%)')
    ax.set_xlabel('Crystal System')
    ax.set_ylabel('Angle Range')
    ax.view_init(elev=25, azim=-60)
    # Keep only bottom XY pane and its grids: hide X/Y panes; keep Z pane; remove vertical (Z-axis) gridlines
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = True
        # remove z-axis gridlines (vertical)
        if hasattr(ax.zaxis, '_axinfo') and 'grid' in ax.zaxis._axinfo:
            ax.zaxis._axinfo['grid']['linewidth'] = 0
    except Exception:
        pass
    plt.tight_layout()
    out_path = os.path.join(root_save_dir, f'{eval_set_name}_accuracy_all_by_cs_3d.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bars_accuracy_all_by_cs_across_angles(root_save_dir, angle_ranges, angle_name_map, eval_set_name, crystal_systems):
    """
    Build a 2D grouped bar chart: x=crystal system, hue=angle range, y=overall HKL accuracy (%).
    Acc for a system is computed as sum(correct)/sum(total) over its space groups.
    Saves as {eval_set_name}_accuracy_all_by_cs_bars.png in root_save_dir.
    """
    systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
    records = []
    for frac in angle_ranges:
        angle_name = angle_name_map[frac]
        subdir = os.path.join(root_save_dir, angle_name)
        stats_path = os.path.join(subdir, f'{eval_set_name}_stats_by_sg.json')
        if not os.path.exists(stats_path):
            continue
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats_by_sg = {int(k): v for k, v in json.load(f).items()}
        for sys_name in systems:
            sg_range = crystal_systems.get(sys_name, [])
            total_correct = 0
            total_points = 0
            for sg_num_1 in sg_range:
                sg_idx_0 = sg_num_1 - 1
                stats = stats_by_sg.get(sg_idx_0, None)
                if stats is None:
                    continue
                total_correct += int(stats.get('all_correct', 0))
                total_points += int(stats.get('total_points', 0))
            if total_points > 0:
                acc = total_correct / total_points * 100.0
            else:
                acc = 0.0
            records.append({'system': sys_name, 'angle': angle_name, 'accuracy': acc})

    if len(records) == 0:
        return

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(
        data=df,
        x='system',
        y='accuracy',
        hue='angle',
        order=systems,
        hue_order=[angle_name_map[frac] for frac in angle_ranges],
        palette=sns.color_palette(),
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_xlabel('Crystal System', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f'Overall HKL Accuracy by Crystal System and Angle ({eval_set_name.capitalize()} Set)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title='Angle Range')
    plt.tight_layout()
    out_path = os.path.join(root_save_dir, f'{eval_set_name}_accuracy_all_by_cs_bars.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_violins_accuracy_all_by_cs_across_angles(root_save_dir, angle_ranges, angle_name_map, eval_set_name, crystal_systems):
    """
    2D grouped violin plot: x is grouped by crystal system; within each group, one violin per angle.
    Color encodes crystal system (consistent across its violins), not angle.
    Accuracies are per-sample overall HKL accuracies sourced from {subdir}/{eval_set_name}_sample_details.json.
    """
    systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
    palette = sns.color_palette()
    sys_to_color = {name: palette[i % len(palette)] for i, name in enumerate(systems)}

    # Gather long-form records
    import json as _json
    records = []
    angle_names = [angle_name_map[frac] for frac in angle_ranges]
    for frac in angle_ranges:
        angle_name = angle_name_map[frac]
        subdir = os.path.join(root_save_dir, angle_name)
        details_path = os.path.join(subdir, f'{eval_set_name}_sample_details.json')
        if not os.path.exists(details_path):
            continue
        with open(details_path, 'r', encoding='utf-8') as f:
            arr = _json.load(f)
        for rec in arr:
            sg_val = int(rec.get('space_group', -1))
            sg_num_1 = sg_val + 1 if 0 <= sg_val < 230 else sg_val
            if sg_num_1 < 1 or sg_num_1 > 230:
                continue
            system_name = None
            for k, rng in crystal_systems.items():
                if sg_num_1 in rng:
                    system_name = k
                    break
            if system_name is None:
                continue
            acc = float(rec.get('accuracy', 0.0))
            if np.isfinite(acc):
                records.append({'system': system_name, 'angle': angle_name, 'accuracy': acc})

    if len(records) == 0:
        return

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    # Composite x category to control colors per system and order within system
    def make_combo(s, a):
        return f"{s}|{a}"
    df['combo'] = df.apply(lambda r: make_combo(r['system'], r['angle']), axis=1)
    combo_order = [make_combo(sys, angle) for sys in systems for angle in angle_names]
    # Palette per combo (color by system)
    combo_colors = [sys_to_color[sys] for sys in systems for _ in angle_names]

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(20, 6))
    ax = sns.violinplot(
        data=df,
        x='combo',
        y='accuracy',
        order=combo_order,
        palette=combo_colors,
        cut=0,
        scale='width',
        inner='quartile',
        bw_adjust=0.4,
        gridsize=256,
        saturation=0.9,
        linewidth=0.7
    )
    # X tick labels as two-line: system on first line, angle below
    xticklabels = []
    for sys in systems:
        for angle in angle_names:
            xticklabels.append(f"{sys}\n{angle}")
    ax.set_xticklabels(xticklabels, rotation=0)
    # Increase spacing by faking extra margins
    ax.set_xlim(-0.6, len(combo_order) - 0.4)
    ax.set_xlabel('Crystal System | Angle', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(f'Overall HKL Accuracy by Crystal System (colored) and Angle ({eval_set_name.capitalize()} Set)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    # Build legend for system colors
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=sys_to_color[s], edgecolor='black', label=s, alpha=0.9) for s in systems]
    ax.legend(handles=handles, title='Crystal System', loc='upper left')
    plt.tight_layout()
    out_path = os.path.join(root_save_dir, f'{eval_set_name}_accuracy_all_by_cs_violins_by_angle.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=None, fixed_density_divisor=None):
    eval_dataset = MillerDataset(
        paths=eval_paths,
        miller_index_offset=miller_index_offset,
        augment_angle=False,
        augment_scale=False,
        debug=args.debug,
        abs_label=args.abs_label,
        norm_scale=args.norm_scale,
        fixed_clip_fraction=fixed_clip_fraction,
        fixed_density_divisor=fixed_density_divisor
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_offset,
        pin_memory=True,
        persistent_workers=False
    )
    return eval_dataset, eval_loader

def main():
    args = get_parser().parse_args()
    
    miller_index_offset = -args.min_hkl
    num_classes = args.max_hkl - args.min_hkl + 1
    print("--- Dynamic Range Parameters ---")
    print(f"  hkl range: [{args.min_hkl}, {args.max_hkl}]")
    print(f"  Calculated Miller Index Offset: {miller_index_offset}")
    print(f"  Calculated num_classes: {num_classes}")
    print("--------------------")

    # Create main save directory (timestamp)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    root_save_dir = os.path.join(args.save_path, f"eval_{timestamp}")
    os.makedirs(root_save_dir, exist_ok=True)
    print(f"--> Evaluation results will be saved in: {root_save_dir}")
    print(f"--> Parameters: {args}")

    # Process dataset paths
    if args.data_paths is not None:
        # Use multiple dataset paths
        eval_paths = [os.path.join(path, args.eval_set) for path in args.data_paths]
        print(f"--> Evaluation dataset paths: {eval_paths}")
    else:
        print("Error: Must specify dataset paths")
        return

    # First build complete data loader to infer input channels
    _, eval_loader_full = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=None)
    
    try:
        _, feats, _, _, _, _ = next(iter(eval_loader_full))
        in_channels = feats.shape[1]
    except StopIteration:
        print("Unable to get data from dataloader, will use default input channels 4.")
        in_channels = 4

    print(f"--> Detected input feature dimensions: {in_channels}, Miller index class count: {num_classes}")
    assert in_channels == 4, f"Input feature dimensions should be 4, but detected {in_channels}"

    # Create model
    model = XRDT(in_channels=in_channels, num_classes=num_classes).cuda()
    print(f"--> Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    
    # Load checkpoint
    print(f"--> Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print("--> Model parameters loaded successfully")
    
    # Create loss function
    criterion = CombinedLoss(miller_weight=1.0, lattice_weight=5.0, sg_weight=0.2).cuda()
    
    # Loop through different angle range evaluations
    angle_ranges = args.angle_eval_ranges
    summary_results = {}
    angle_name_map = {}
    for frac in angle_ranges:
        name = f'{int(frac*360)}deg'

        angle_name_map[frac] = name
        subdir = os.path.join(root_save_dir, name)
        os.makedirs(subdir, exist_ok=True)

        # Build data loader for this angle
        _, eval_loader = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=frac, fixed_density_divisor=None)

        print(f"--> Starting evaluation of {args.eval_set} set, angle range: {name} ({frac*360:.2f}°)...")
        eval_metrics, eval_extras = evaluate(
            eval_loader,
            model,
            criterion,
            subdir,
            eval_set_name=args.eval_set,
            canonicalize_hkl=args.canonicalize_hkl,
            miller_index_offset=miller_index_offset,
        )

        # Print evaluation results
        print("-" * 80)
        print(f"Evaluation Results ({args.eval_set.upper()} Set, {name}):")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  L_miller: {eval_metrics['loss_miller']:.4f}")
        print(f"  L_lattice: {eval_metrics['loss_lattice']:.4f}")
        print(f"  L_sg: {eval_metrics['loss_sg']:.4f}")
        print(f"  Miller Acc: {eval_metrics['acc_all']:.2f}%")
        print(f"  H Acc: {eval_metrics['acc_h']:.2f}%")
        print(f"  K Acc: {eval_metrics['acc_k']:.2f}%")
        print(f"  L Acc: {eval_metrics['acc_l']:.2f}%")
        print(f"  SG Acc: {eval_metrics['sg_acc']:.2f}%")
        print(f"  Lattice MAE A: {eval_metrics['lattice_mae_a']:.4f}")
        print(f"  Lattice MAE Ang: {eval_metrics['lattice_mae_ang']:.4f}")
        print("-" * 80)

        # Save to subdirectory JSON
        results_path = os.path.join(subdir, f'{args.eval_set}_evaluation_results.json')
        serializable_metrics = {}
        for key, value in eval_metrics.items():
            if hasattr(value, 'item'):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        summary_results[name] = serializable_metrics
        # Additionally save data for overlay plotting
        with open(os.path.join(subdir, f'{args.eval_set}_extras.npy'), 'wb') as f:
            np.save(f, {
                'abc_errors': np.array(eval_extras['abc_errors'], dtype=np.float32),
                'ang_errors': np.array(eval_extras['ang_errors'], dtype=np.float32),
            }, allow_pickle=True)
        # Save stats_by_sg as json (numerized)
        stats_json = {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in eval_extras['stats_by_sg'].items()}
        with open(os.path.join(subdir, f'{args.eval_set}_stats_by_sg.json'), 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, indent=2, ensure_ascii=False)

        # Clean up data loader
        del eval_loader
        torch.cuda.empty_cache()

    # Extra noisy evaluation on full-angle data
    if bool(getattr(args, 'noisy_eval_enable', True)):
        print("--> Running extra NOISY evaluation on full-angle data")
        subdir = os.path.join(root_save_dir, 'noisy')
        os.makedirs(subdir, exist_ok=True)
        # Build full-angle loader
        _, eval_loader_noisy = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=1.0)
        noisy_metrics, _ = evaluate(
            eval_loader_noisy,
            model,
            criterion,
            subdir,
            eval_set_name=args.eval_set,
            canonicalize_hkl=args.canonicalize_hkl,
            miller_index_offset=miller_index_offset,
            apply_noise=True,
            noise_prob=float(getattr(args, 'noise_eval_prob', 0.05)),
            noise_micro=float(getattr(args, 'noise_eval_micro', 0.01)),
        )
        with open(os.path.join(subdir, f'{args.eval_set}_evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump({k: (v.item() if hasattr(v, 'item') else v) for k, v in noisy_metrics.items()}, f, indent=2, ensure_ascii=False)
        del eval_loader_noisy
        torch.cuda.empty_cache()

    # Write overall summary to main directory
    with open(os.path.join(root_save_dir, f'summary_{args.eval_set}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    # Overlay plotting: lattice cumulative distribution & h/k/l accuracy
    try:
        overlay_lattice_path = os.path.join(root_save_dir, f'{args.eval_set}_lattice_cumulative_distribution.png')
        overlay_acc_paths = {
            'h': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_h_by_sg.png'),
            'k': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_k_by_sg.png'),
            'l': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_l_by_sg.png'),
            'all': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_all_by_sg.png'),
        }

        # Load data from subdirectories
        ordered_items = []
        for frac in angle_ranges:
            name = angle_name_map[frac]
            subdir = os.path.join(root_save_dir, name)
            extras_npy = os.path.join(subdir, f'{args.eval_set}_extras.npy')
            stats_json_path = os.path.join(subdir, f'{args.eval_set}_stats_by_sg.json')
            abc_errors = None
            ang_errors = None
            if os.path.exists(extras_npy):
                obj = np.load(extras_npy, allow_pickle=True).item()
                abc_errors = obj.get('abc_errors', None)
                ang_errors = obj.get('ang_errors', None)
            stats_by_sg = None
            if os.path.exists(stats_json_path):
                with open(stats_json_path, 'r', encoding='utf-8') as f:
                    stats_by_sg = {int(k): v for k, v in json.load(f).items()}
            ordered_items.append((name, abc_errors, ang_errors, stats_by_sg))

        # Plot overlay lattice cumulative distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx, (name, abc_errors, ang_errors, _) in enumerate(ordered_items):
            if abc_errors is not None and len(abc_errors) > 0:
                abc_errors = np.array(abc_errors)
                abc_thresholds = np.logspace(-2, 0, 400)
                abc_cumulative = [(abc_errors <= t).sum() / len(abc_errors) * 100 for t in abc_thresholds]
                ax1.plot(abc_thresholds, abc_cumulative, '-', linewidth=2, label=name, color=color_cycle[idx % len(color_cycle)])
            if ang_errors is not None and len(ang_errors) > 0:
                ang_errors = np.array(ang_errors)
                ang_thresholds = np.logspace(-1, 1, 400)
                ang_cumulative = [(ang_errors <= t).sum() / len(ang_errors) * 100 for t in ang_thresholds]
                ax2.plot(ang_thresholds, ang_cumulative, '-', linewidth=2, label=name, color=color_cycle[idx % len(color_cycle)])
        ax1.set_xscale('log'); ax2.set_xscale('log')
        ax1.set_xlabel('Absolute Error in ABC (Å)'); ax2.set_xlabel('Absolute Error in Angles (degrees)')
        ax1.set_ylabel('Cumulative Percentage (%)'); ax2.set_ylabel('Cumulative Percentage (%)')
        ax1.set_title(f'Cell Length Accuracy Distribution Overlay\n({args.eval_set.capitalize()} Set)')
        ax2.set_title(f'Cell Angles Accuracy Distribution Overlay\n({args.eval_set.capitalize()} Set)')
        ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
        ax1.legend(); ax2.legend()
        plt.tight_layout()
        plt.savefig(overlay_lattice_path, dpi=300, bbox_inches='tight')
        plt.close()
        # Also save a copy of angle-sensitivity lattice CDF into root directory
        try:
            import shutil
            overlay_copy = os.path.join(root_save_dir, f'{args.eval_set}_angle_lattice_cumulative_distribution.png')
            shutil.copyfile(overlay_lattice_path, overlay_copy)
        except Exception:
            pass

        # Overlay h/k/l/all scatter plots
        crystal_systems = {
            'triclinic': list(range(1, 3)), 'monoclinic': list(range(3, 16)),
            'orthorhombic': list(range(16, 75)), 'tetragonal': list(range(75, 143)),
            'trigonal': list(range(143, 168)), 'hexagonal': list(range(168, 195)),
            'cubic': list(range(195, 231))
        }

        def build_series(stats_by_sg, key):
            series = []
            sizes = []
            for sg_idx in range(230):
                stats = stats_by_sg.get(sg_idx, {}) if stats_by_sg is not None else {}
                total_points = stats.get('total_points', 0)
                correct_points = stats.get('all_correct' if key == 'all' else f'{key}_correct', 0)
                acc = (correct_points / total_points * 100) if total_points > 0 else 0.0
                series.append(acc)
                # Set point size based on data point count
                point_size = max(20, min(100, total_points / 10)) if total_points > 0 else 20
                sizes.append(point_size)
            return np.array(series, dtype=np.float32), np.array(sizes, dtype=np.float32)

        # Assign stable colors for different angle ranges
        angle_names = [name for name, _, __, ___ in ordered_items]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {angle_names[i]: color_cycle[i % len(color_cycle)] for i in range(len(angle_names))}

        for key in ['h', 'k', 'l', 'all']:
            plt.figure(figsize=(20, 8))
            x = np.arange(1, 231)

            # Build series and sizes for each angle first
            series_by_angle = {}
            sizes_by_angle = {}
            for name, _, __, stats_by_sg in ordered_items:
                series_by_angle[name], sizes_by_angle[name] = build_series(stats_by_sg, key)

            # Plot scatter for each angle range
            for name in angle_names:
                series = series_by_angle[name]
                sizes = sizes_by_angle[name]
                # Only plot points with data
                mask = series > 0
                if np.any(mask):
                    plt.scatter(x[mask], series[mask], s=sizes[mask], 
                              c=color_map[name], alpha=0.7, edgecolors='black', 
                              linewidth=0.5, label=name)

            # plt.xlabel('Space Group Number', fontsize=12)
            # plt.ylabel('Accuracy (%)', fontsize=12)
            # plt.title(f'{key.upper()} Accuracy by Space Group (Overlay Points)', fontsize=14)
            plt.ylim(0, 100)
            plt.xlim(0, 231)
            
            # Add grid
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Set x-axis ticks
            tick_positions = np.arange(0, 231, 20)
            plt.xticks(tick_positions)

            # Build legend
            plt.legend(loc='upper right', title='Angle Ranges')
            plt.tight_layout()
            plt.savefig(overlay_acc_paths[key], dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Failed to generate overlay plots: {e}")

    # 3D violin across angles (by crystal system)
    try:
        # Reconstruct crystal systems mapping here for safety
        crystal_systems = {
            'triclinic': list(range(1, 3)), 'monoclinic': list(range(3, 16)),
            'orthorhombic': list(range(16, 75)), 'tetragonal': list(range(75, 143)),
            'trigonal': list(range(143, 168)), 'hexagonal': list(range(168, 195)),
            'cubic': list(range(195, 231))
        }
        # plot 2D grouped violins by angle per crystal system
        plot_violins_accuracy_all_by_cs_across_angles(root_save_dir, angle_ranges, angle_name_map, args.eval_set, crystal_systems)
        # copy per-angle lattice CDF overlay as an explicit angle sensitivity artifact
        try:
            import shutil
            src = os.path.join(root_save_dir, f'{args.eval_set}_lattice_cumulative_distribution.png')
            dst = os.path.join(root_save_dir, f'{args.eval_set}_angle_lattice_cumulative_distribution.png')
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to generate grouped bars by cs: {e}")

    # Density sensitivity study on full-angle data
    try:
        density_divs = [int(x) for x in getattr(args, 'density_eval_ranges', []) if int(x) >= 1]
        if len(density_divs) > 0:
            crystal_systems = {
                'triclinic': list(range(1, 3)), 'monoclinic': list(range(3, 16)),
                'orthorhombic': list(range(16, 75)), 'tetragonal': list(range(75, 143)),
                'trigonal': list(range(143, 168)), 'hexagonal': list(range(168, 195)),
                'cubic': list(range(195, 231))
            }
            # Run evaluations per density divisor on full-angle data (no clipping)
            for div in density_divs:
                dname = f'density_1_{div}'
                subdir = os.path.join(root_save_dir, dname)
                os.makedirs(subdir, exist_ok=True)
                _, dloader = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=None, fixed_density_divisor=div)
                print(f"--> Starting evaluation of {args.eval_set} set, density: {dname} (1/{div})...")
                d_metrics, d_extras = evaluate(
                    dloader,
                    model,
                    criterion,
                    subdir,
                    eval_set_name=args.eval_set,
                    canonicalize_hkl=args.canonicalize_hkl,
                    miller_index_offset=miller_index_offset,
                )
                # Save density extras for overlay plotting
                try:
                    with open(os.path.join(subdir, f'{args.eval_set}_extras.npy'), 'wb') as f:
                        np.save(f, {
                            'abc_errors': np.array(d_extras.get('abc_errors', []), dtype=np.float32),
                            'ang_errors': np.array(d_extras.get('ang_errors', []), dtype=np.float32),
                        }, allow_pickle=True)
                except Exception:
                    pass
                del dloader
                torch.cuda.empty_cache()

            # Extra NOISY run on full-angle for density context (saved under 'noisy') is already executed above

            # Build density-based grouped violins by crystal system
            def _plot_violins_by_density(root_dir, density_divs, eval_set_name, crystal_systems):
                import json as _json
                systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
                palette = sns.color_palette()
                sys_to_color = {name: palette[i % len(palette)] for i, name in enumerate(systems)}
                records = []
                for div in density_divs:
                    dname = f'density_1_{div}'
                    subdir = os.path.join(root_dir, dname)
                    details_path = os.path.join(subdir, f'{eval_set_name}_sample_details.json')
                    if not os.path.exists(details_path):
                        continue
                    with open(details_path, 'r', encoding='utf-8') as f:
                        arr = _json.load(f)
                    for rec in arr:
                        sg_val = int(rec.get('space_group', -1))
                        sg_num_1 = sg_val + 1 if 0 <= sg_val < 230 else sg_val
                        if sg_num_1 < 1 or sg_num_1 > 230:
                            continue
                        system_name = None
                        for k, rng in crystal_systems.items():
                            if sg_num_1 in rng:
                                system_name = k
                                break
                        if system_name is None:
                            continue
                        acc = float(rec.get('accuracy', 0.0))
                        if np.isfinite(acc):
                            records.append({'system': system_name, 'density': f'1/{div}', 'accuracy': acc})

                if len(records) == 0:
                    return

                import pandas as pd
                df = pd.DataFrame.from_records(records)
                def make_combo(s, a):
                    return f"{s}|{a}"
                df['combo'] = df.apply(lambda r: make_combo(r['system'], r['density']), axis=1)
                combo_order = [make_combo(sys, f'1/{d}') for sys in systems for d in density_divs]
                combo_colors = [sys_to_color[sys] for sys in systems for _ in density_divs]

                sns.set_theme(style='whitegrid')
                plt.figure(figsize=(20, 6))
                ax = sns.violinplot(
                    data=df,
                    x='combo',
                    y='accuracy',
                    order=combo_order,
                    palette=combo_colors,
                    cut=0,
                    scale='width',
                    inner='quartile',
                    bw_adjust=0.4,
                    gridsize=256,
                    saturation=0.9,
                    linewidth=0.7
                )
                xticklabels = []
                for sys in systems:
                    for d in density_divs:
                        xticklabels.append(f"{sys}\n1/{d}")
                ax.set_xticklabels(xticklabels, rotation=0)
                # widen group spacing
                ax.set_xlim(-0.6, len(combo_order) - 0.4)
                ax.set_xlabel('Crystal System | Density', fontsize=12)
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_ylim(0, 100)
                ax.set_title(f'Overall HKL Accuracy by Crystal System (colored) and Density ({eval_set_name.capitalize()} Set)', fontsize=14)
                ax.grid(True, axis='y', alpha=0.3)
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=sys_to_color[s], edgecolor='black', label=s, alpha=0.9) for s in systems]
                ax.legend(handles=handles, title='Crystal System', loc='upper right')
                plt.tight_layout()
                out_path = os.path.join(root_dir, f'{eval_set_name}_accuracy_all_by_cs_across_density.png')
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()

            _plot_violins_by_density(root_save_dir, density_divs, args.eval_set, crystal_systems)
            # Density overlay lattice cumulative distribution
            try:
                density_overlay_path = os.path.join(root_save_dir, f'{args.eval_set}_density_lattice_cumulative_distribution.png')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for idx, div in enumerate(density_divs):
                    dname = f'density_1_{div}'
                    subdir = os.path.join(root_save_dir, dname)
                    extras_npy = os.path.join(subdir, f'{args.eval_set}_extras.npy')
                    label = f'1/{div}'
                    if os.path.exists(extras_npy):
                        obj = np.load(extras_npy, allow_pickle=True).item()
                        abc_errors = obj.get('abc_errors', None)
                        ang_errors = obj.get('ang_errors', None)
                        if abc_errors is not None and len(abc_errors) > 0:
                            abc_errors = np.array(abc_errors)
                            abc_thresholds = np.logspace(-2, 0, 400)
                            abc_cumulative = [(abc_errors <= t).sum() / len(abc_errors) * 100 for t in abc_thresholds]
                            ax1.plot(abc_thresholds, abc_cumulative, '-', linewidth=2, label=label, color=color_cycle[idx % len(color_cycle)])
                        if ang_errors is not None and len(ang_errors) > 0:
                            ang_errors = np.array(ang_errors)
                            ang_thresholds = np.logspace(-1, 1, 400)
                            ang_cumulative = [(ang_errors <= t).sum() / len(ang_errors) * 100 for t in ang_thresholds]
                            ax2.plot(ang_thresholds, ang_cumulative, '-', linewidth=2, label=label, color=color_cycle[idx % len(color_cycle)])
                ax1.set_xscale('log'); ax2.set_xscale('log')
                ax1.set_xlabel('Absolute Error in ABC (Å)'); ax2.set_xlabel('Absolute Error in Angles (degrees)')
                ax1.set_ylabel('Cumulative Percentage (%)'); ax2.set_ylabel('Cumulative Percentage (%)')
                ax1.set_title(f'Cell Length Accuracy Distribution Overlay (Density)\n({args.eval_set.capitalize()} Set)')
                ax2.set_title(f'Cell Angles Accuracy Distribution Overlay (Density)\n({args.eval_set.capitalize()} Set)')
                ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
                ax1.legend(); ax2.legend()
                plt.tight_layout()
                plt.savefig(density_overlay_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to run density study: {e}")

    # Confusion matrix for SG on full-angle data (if available)
    try:
        def _plot_confusion_matrix_sg(root_save_dir, eval_set_name, subdir_angle_name='360deg'):
            import json as _json
            details_path = os.path.join(root_save_dir, subdir_angle_name, f'{eval_set_name}_sample_details.json')
            if not os.path.exists(details_path):
                return
            with open(details_path, 'r', encoding='utf-8') as f:
                arr = _json.load(f)
            y_true = []
            y_pred = []
            for rec in arr:
                tgt = int(rec.get('space_group', -1))
                pred = int(rec.get('pred_sg', -1))
                if tgt >= 0 and pred >= 0:
                    y_true.append(tgt)
                    y_pred.append(pred)
            if len(y_true) == 0:
                return
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(range(230)))
            cm_norm = cm.astype(np.float32)
            row_sums = cm_norm.sum(axis=1, keepdims=True) + 1e-8
            cm_norm = cm_norm / row_sums * 100.0
            sns.set_theme(style='white')
            plt.figure(figsize=(18, 14))
            ax = sns.heatmap(cm_norm, cmap='viridis', cbar_kws={'label': 'Row-Normalized (%)'}, vmin=0, vmax=100)
            ax.set_xlabel('Predicted SG (0-based)')
            ax.set_ylabel('True SG (0-based)')
            ax.set_title(f'Space Group Confusion Matrix ({eval_set_name.upper()} - {subdir_angle_name})')
            ax.set_xticks(list(range(0, 230, 20)))
            ax.set_yticks(list(range(0, 230, 20)))
            plt.tight_layout()
            out_path = os.path.join(root_save_dir, f'{eval_set_name}_sg_confusion_{subdir_angle_name}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
        _plot_confusion_matrix_sg(root_save_dir, args.eval_set, '360deg')
    except Exception as e:
        print(f"Failed to draw SG confusion matrix: {e}")

    print("Evaluation completed.")

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
