import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import contextlib
import torch.optim as optim
from torch.utils.data import DataLoader

from RCT.dataset import MillerDataset, collate_fn_offset
from eval import evaluate as eval_evaluate


def log_train_scalars(writer, loss_value, loss_dict, lr, global_step, clip_lower_now=None, angle_range_now=None, density_clip_prob_now=None, noise_adding_prob_now=None):
    writer.add_scalar('Train/Loss_Total', loss_value, global_step)
    writer.add_scalar('Train/Loss_Miller', loss_dict['loss_miller'].detach().item(), global_step)
    writer.add_scalar('Train/Loss_h', loss_dict['loss_h'].detach().item(), global_step)
    writer.add_scalar('Train/Loss_k', loss_dict['loss_k'].detach().item(), global_step)
    writer.add_scalar('Train/Loss_l', loss_dict['loss_l'].detach().item(), global_step)
    writer.add_scalar('Train/Loss_Lattice', loss_dict['loss_lattice'].detach().item(), global_step)
    writer.add_scalar('Train/Loss_SG', loss_dict['loss_sg'].detach().item(), global_step)
    writer.add_scalar('Misc/LR', lr, global_step)
    if clip_lower_now is not None:
        writer.add_scalar('Misc/ClipLowerNow', clip_lower_now, global_step)
    if angle_range_now is not None:
        writer.add_scalar('Misc/AngleRange', angle_range_now, global_step)
    if density_clip_prob_now is not None:
        writer.add_scalar('Misc/DensityClipProb', density_clip_prob_now, global_step)
    if noise_adding_prob_now is not None:
        writer.add_scalar('Misc/NoiseAddingProb', noise_adding_prob_now, global_step)


def log_val_metrics(writer, metrics, epoch: int):
    writer.add_scalar('Val/Loss', metrics['loss'], epoch)
    writer.add_scalar('Val/L_miller', metrics['loss_miller'], epoch)
    writer.add_scalar('Val/L_lattice', metrics['loss_lattice'], epoch)
    writer.add_scalar('Val/L_sg', metrics['loss_sg'], epoch)
    writer.add_scalar('Val/Miller_Accuracy_All', metrics['acc_all'], epoch)
    writer.add_scalar('Val/Miller_Accuracy_h', metrics['acc_h'], epoch)
    writer.add_scalar('Val/Miller_Accuracy_k', metrics['acc_k'], epoch)
    writer.add_scalar('Val/Miller_Accuracy_l', metrics['acc_l'], epoch)
    writer.add_scalar('Val/SG_Accuracy', metrics['sg_acc'], epoch)
    writer.add_scalar('Val/Lattice_MAE_A', metrics['lattice_mae_a'], epoch)
    writer.add_scalar('Val/Lattice_MAE_Ang', metrics['lattice_mae_ang'], epoch)


def draw_overlay_plots(ordered_items, full_root, eval_set_name='val'):
    overlay_lattice_path = os.path.join(full_root, f'{eval_set_name}_lattice_cumulative_distribution.png')
    overlay_acc_paths = {
        'h': os.path.join(full_root, f'{eval_set_name}_accuracy_h_by_sg.png'),
        'k': os.path.join(full_root, f'{eval_set_name}_accuracy_k_by_sg.png'),
        'l': os.path.join(full_root, f'{eval_set_name}_accuracy_l_by_sg.png'),
        'all': os.path.join(full_root, f'{eval_set_name}_accuracy_all_by_sg.png'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, (name, abc_errors, ang_errors, _) in enumerate(ordered_items):
        if abc_errors is not None and len(abc_errors) > 0:
            if hasattr(abc_errors, 'detach') and hasattr(abc_errors, 'cpu'):
                abc_errors = abc_errors.detach().cpu().numpy()
            elif isinstance(abc_errors, (list, tuple)) and len(abc_errors) > 0 and hasattr(abc_errors[0], 'item'):
                abc_errors = np.array([x.item() if hasattr(x, 'item') else float(x) for x in abc_errors], dtype=np.float32)
            else:
                abc_errors = np.array(abc_errors)
            abc_thresholds = np.logspace(-2, 0, 400)
            abc_cumulative = [(abc_errors <= t).sum() / len(abc_errors) * 100 for t in abc_thresholds]
            ax1.plot(abc_thresholds, abc_cumulative, '-', linewidth=2, label=name, color=color_cycle[idx % len(color_cycle)])
        if ang_errors is not None and len(ang_errors) > 0:
            if hasattr(ang_errors, 'detach') and hasattr(ang_errors, 'cpu'):
                ang_errors = ang_errors.detach().cpu().numpy()
            elif isinstance(ang_errors, (list, tuple)) and len(ang_errors) > 0 and hasattr(ang_errors[0], 'item'):
                ang_errors = np.array([x.item() if hasattr(x, 'item') else float(x) for x in ang_errors], dtype=np.float32)
            else:
                ang_errors = np.array(ang_errors)
            ang_thresholds = np.logspace(-1, 1, 400)
            ang_cumulative = [(ang_errors <= t).sum() / len(ang_errors) * 100 for t in ang_thresholds]
            ax2.plot(ang_thresholds, ang_cumulative, '-', linewidth=2, label=name, color=color_cycle[idx % len(color_cycle)])
    ax1.set_xscale('log'); ax2.set_xscale('log')
    ax1.set_xlabel('Absolute Error in ABC (Å)'); ax2.set_xlabel('Absolute Error in Angles (degrees)')
    ax1.set_ylabel('Cumulative Percentage (%)'); ax2.set_ylabel('Cumulative Percentage (%)')
    ax1.set_title('Cell Length Accuracy Distribution Overlay\n(Val Set)')
    ax2.set_title('Cell Angles Accuracy Distribution Overlay\n(Val Set)')
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    ax1.legend(); ax2.legend()
    plt.tight_layout()
    plt.savefig(overlay_lattice_path, dpi=300, bbox_inches='tight')
    try:
        base, _ = os.path.splitext(overlay_lattice_path)
        plt.savefig(base + '.eps', format='eps', bbox_inches='tight')
    except Exception:
        pass
    plt.close()

    angle_names = [name for name, _, __, ___ in ordered_items]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {angle_names[i]: color_cycle[i % len(color_cycle)] for i in range(len(angle_names))}

    def build_series(stats_by_sg, key):
        series = []
        for sg_idx in range(230):
            stats = stats_by_sg.get(sg_idx, {}) if stats_by_sg is not None else {}
            total_points = stats.get('total_points', 0)
            correct_points = stats.get('all_correct' if key == 'all' else f'{key}_correct', 0)
            if hasattr(correct_points, 'item'):
                correct_points = correct_points.item()
            if hasattr(total_points, 'item'):
                total_points = total_points.item()
            acc = (correct_points / total_points * 100) if total_points > 0 else 0.0
            series.append(acc)
        return np.array(series, dtype=np.float32)

    for key in ['h', 'k', 'l', 'all']:
        plt.figure(figsize=(20, 6))
        x = np.arange(1, 231)
        series_by_angle = {name: build_series(stats_by_sg, key) for name, _, __, stats_by_sg in ordered_items}
        for sg_idx in range(230):
            values = [(name, float(series_by_angle[name][sg_idx])) for name in angle_names]
            values_sorted = sorted(values, key=lambda t: t[1], reverse=True)
            for order_idx, (name, height) in enumerate(values_sorted):
                if height <= 0:
                    continue
                plt.bar(x[sg_idx], height, width=0.8, color=color_map[name], edgecolor='none', alpha=0.8, zorder=order_idx)
        plt.xlabel('Space Group Number')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{key.upper()} Accuracy by Space Group (Overlay Bars)')
        plt.ylim(0, 100)
        tick_positions = np.arange(1, 231, 10)
        plt.xticks(tick_positions)
        legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[name], alpha=0.8) for name in angle_names]
        plt.legend(legend_handles, angle_names, loc='upper right')
        plt.tight_layout()
        plt.savefig(overlay_acc_paths[key], dpi=300, bbox_inches='tight')
        try:
            base, _ = os.path.splitext(overlay_acc_paths[key])
            plt.savefig(base + '.eps', format='eps', bbox_inches='tight')
        except Exception:
            pass
        plt.close()


# -------------------------------
# Data/build helpers
# -------------------------------
def build_paths(args):
    train_paths = [os.path.join(path, 'train') for path in args.data_paths]
    val_paths = [os.path.join(path, 'val') for path in args.data_paths]
    return train_paths, val_paths


def build_datasets(train_paths, val_paths, miller_index_offset, args):
    train_dataset = MillerDataset(
        paths=train_paths,
        miller_index_offset=miller_index_offset,
        augment_angle=args.augment_angle,
        augment_scale=False,
        debug=args.debug,
        norm_scale=args.norm_scale,
        lattice_stats_json=getattr(args, 'lattice_stats_json', None),
        intensity_mean=getattr(args, 'intensity_mean', None),
        intensity_std=getattr(args, 'intensity_std', None)
    )
    val_dataset = MillerDataset(
        paths=val_paths,
        miller_index_offset=miller_index_offset,
        augment_angle=False,
        augment_scale=False,
        debug=0,
        norm_scale=args.norm_scale,
        lattice_stats_json=getattr(args, 'lattice_stats_json', None),
        intensity_mean=getattr(args, 'intensity_mean', None),
        intensity_std=getattr(args, 'intensity_std', None)
    )
    return train_dataset, val_dataset


def build_train_loader(train_dataset, args):
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_offset,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )


def build_eval_loader(paths, miller_index_offset, args, fixed_clip_fraction=None):
    dataset = MillerDataset(
        paths=paths,
        miller_index_offset=miller_index_offset,
        augment_angle=False,
        augment_scale=False,
        debug=0,
        norm_scale=args.norm_scale,
        fixed_clip_fraction=fixed_clip_fraction,
        lattice_stats_json=getattr(args, 'lattice_stats_json', None),
        intensity_mean=getattr(args, 'intensity_mean', None),
        intensity_std=getattr(args, 'intensity_std', None)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn_offset,
        pin_memory=True,
        persistent_workers=False
    )
    return dataset, loader


# -------------------------------
# Scheduler helpers
# -------------------------------
def create_scheduler(optimizer, args, steps_per_epoch: int):
    if args.warmup_epochs > 0:
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_epochs / args.epochs,
            anneal_strategy=args.warmup_method,
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)


# -------------------------------
# Evaluation helpers
# -------------------------------
def run_fast_eval(val_paths, miller_index_offset, args, model, criterion):
    _, val_loader_simple = build_eval_loader(val_paths, miller_index_offset, args, fixed_clip_fraction=None)
    val_metrics, _ = eval_evaluate(val_loader_simple, model, criterion, save_path=None, eval_set_name='val', noise_prob=0.0, noise_micro=0.001)
    del val_loader_simple
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return val_metrics


def run_full_eval(val_paths, miller_index_offset, args, model, criterion, save_root):
    angle_eval_ranges = [1.0, 0.5, 0.1667, 0.0278]
    ordered_items = []
    for frac in angle_eval_ranges:
        name = f"{int(frac*360)}deg"
        subdir = os.path.join(save_root, name)
        os.makedirs(subdir, exist_ok=True)
        _, vloader = build_eval_loader(val_paths, miller_index_offset, args, fixed_clip_fraction=frac)
        metrics, extras = eval_evaluate(vloader, model, criterion, save_path=subdir, eval_set_name='val')
        ordered_items.append((name, extras.get('abc_errors', []), extras.get('ang_errors', []), extras.get('stats_by_sg', {})))
        with open(os.path.join(subdir, 'val_evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump({k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)
        del vloader
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    draw_overlay_plots(ordered_items, save_root, eval_set_name='val')


