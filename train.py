import argparse
import os
import time
from datetime import datetime
import gc
import contextlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from RCT.model import RCT
from RCT.dataset import MillerDataset, collate_fn_offset
from RCT.loss import CombinedLoss
import warnings
from math import pi, cos
warnings.filterwarnings("ignore")

import multiprocessing as mp
from RCT.utils import (
    log_train_scalars,
    log_val_metrics,
    build_paths,
    build_datasets,
    build_train_loader,
    create_scheduler,
    run_fast_eval,
    run_full_eval,
)

def get_parser():
    parser = argparse.ArgumentParser(description='RCT')
    parser.add_argument('--model', type=str, default='RCT')
    parser.add_argument('--device', type=str,   default='cuda', choices=['cpu', 'cuda'], help='device to use for training')
    parser.add_argument('--min_hkl', type=int, default=-5)
    parser.add_argument('--max_hkl', type=int, default=5)
    parser.add_argument('--data_paths', nargs='+', help='paths to dataset directories')
    parser.add_argument('--save_path', type=str, default=None, help='path to save training results')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path, only load model weights')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path, load full training state')
    parser.add_argument('--debug', type=int, default=0, help='if > 0, limit training set size for fast debugging')
    parser.add_argument('--batch_size', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--log_freq', type=int, default=10, help='tensorboard logging frequency (iterations)')
    parser.add_argument('--full_eval_freq', type=int, default=100, help='full evaluation frequency (epochs)')
    
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--loss_weights', type=float, default=[1.0, 1.0, 0.2], nargs=3, help='[L_miller, L_lattice, L_sg]')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--warmup_method', type=str, default='cos', choices=['linear', 'cos'])

    parser.add_argument('--augment_angle', type=bool, default=True, help='enable angle augmentation')
    parser.add_argument('--norm_scale', type=bool, default=True, help='enable coordinate normalization')
    # dynamic angle clipping
    parser.add_argument('--dynamic_angle_clip', type=bool, default=True, help='enable dynamic angle clipping')
    parser.add_argument('--angle_clip_schedule', type=str, default='cos', choices=['linear', 'cos'], help='dynamic angle clipping method')
    parser.add_argument('--angle_range_low', type=float, default=0.083334, help='angle clipping lower bound')
    # dynamic density clipping (probabilistic angular downsampling per sample)
    parser.add_argument('--dynamic_density_clip', type=bool, default=True, help='enable dynamic density downsampling along angle')
    parser.add_argument('--density_clip_schedule', type=str, default='cos', choices=['linear', 'cos'], help='schedule for density clip probability growth')
    parser.add_argument('--density_clip_prob_high', type=float, default=0.25, help='upper bound probability to apply density clip to a sample')
    # dynamic noise adding
    parser.add_argument('--dynamic_noise_adding', type=bool, default=True, help='enable dynamic noise adding (only process points of lower 10% intensity)')
    parser.add_argument('--noise_adding_schedule', type=str, default='cos', choices=['linear', 'cos'], help='schedule for noise adding probability growth')
    parser.add_argument('--noise_adding_prob_high', type=float, default=0.1, help='upper bound probability to apply noise adding to lower 10% points')
    # intensity standardization
    parser.add_argument('--lattice_stats_json', type=str, default='cell_params_statistics.json', help='Path to lattice stats JSON for standardization (from analyze script)')
    parser.add_argument('--intensity_mean', type=float, default=7.029995, help='Mean of intensity for standardization; if set with std, use (I-mean)/std')
    parser.add_argument('--intensity_std', type=float, default=4.295749, help='Std of intensity for standardization; must be > 0')
    # sample-accuracy-based Miller loss gating
    parser.add_argument('--gate_by_sample_acc', type=bool, default=False, help='Enable gating Miller loss by per-sample HKL accuracy')
    parser.add_argument('--gate_start_epoch', type=int, default=50, help='Start gating after this epoch (strictly >)')
    parser.add_argument('--gate_sample_acc_threshold', type=float, default=0.01, help='Per-sample overall HKL accuracy threshold in [0,1]')
    return parser

def load_pretrained_flex(model: torch.nn.Module, ckpt_path: str):
    print(f"--> Load pretrained weights from '{ckpt_path}'")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_state = model.state_dict()

    def maybe_strip_module(k: str) -> str:
        if k.startswith('module.') and not any(s.startswith('module.') for s in model_state.keys()):
            return k[len('module.'):]
        return k

    filtered = {}
    loaded_keys = []
    skipped_keys = []

    for k, v in state_dict.items():
        k2 = maybe_strip_module(k)
        if k2 in model_state and hasattr(v, 'shape') and model_state[k2].shape == v.shape:
            filtered[k2] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    missing_in_ckpt = [k for k in model_state.keys() if k not in filtered]

    msg = (
        f"--> Pretrained load summary | loaded: {len(loaded_keys)} | "
        f"skipped(shape/key): {len(skipped_keys)} | missing_in_ckpt: {len(missing_in_ckpt)}"
    )
    print(msg)
    if len(skipped_keys) > 0:
        preview = ', '.join(skipped_keys[:10])
        more = '' if len(skipped_keys) <= 10 else f" (+{len(skipped_keys)-10} more)"
        print(f"--> Skipped keys (preview): {preview}{more}")

    model.load_state_dict(filtered, strict=False)
    return ckpt

def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, 'last_model.pth'))
    if is_best:
        torch.save(state, os.path.join(save_path, 'best_model.pth'))

def compute_clip_lower(global_iter: int, total_iters: int, start_lower: float, end_lower: float, method: str) -> float:
    if total_iters <= 1:
        return end_lower
    t = min(max(global_iter / (total_iters - 1), 0.0), 1.0)
    if method == 'linear':
        return float(start_lower + (end_lower - start_lower) * t)
    elif method == 'cos':
        decay = 0.5 * (1.0 + cos(pi * t))
        return float(end_lower + (start_lower - end_lower) * decay)
    else:
        raise ValueError(f"Invalid method: {method}")

def apply_angle_clipping_batch(coords, feats, labels, offsets, clip_lower: float):
    device = feats.device
    dtype_offsets = offsets.dtype
    new_coords_list = []
    new_feats_list = []
    new_labels_list = []
    new_lengths = []

    batch_size = int(offsets.numel())
    has_many = batch_size > 1
    start_idx = 0
    for b in range(batch_size):
        end_idx = int(offsets[b].item())
        phi = feats[start_idx:end_idx, 0]
        if has_many:
            ratio = float(b) / float(batch_size - 1)
            frac = clip_lower + (1.0 - clip_lower) * ratio
        else:
            frac = 1.0
        mask = phi <= frac
        if mask.sum().item() == 0:
            local_min = torch.argmin(phi)
            mask = torch.zeros_like(phi, dtype=torch.bool)
            mask[local_min] = True
        new_coords_list.append(coords[start_idx:end_idx][mask])
        new_feats_list.append(feats[start_idx:end_idx][mask])
        new_labels_list.append(labels[start_idx:end_idx][mask])
        new_lengths.append(int(mask.sum().item()))
        start_idx = end_idx

    new_coords = torch.cat(new_coords_list, dim=0).contiguous()
    new_feats = torch.cat(new_feats_list, dim=0).contiguous()
    new_labels = torch.cat(new_labels_list, dim=0).contiguous()
    new_offsets = torch.tensor(new_lengths, dtype=dtype_offsets, device=offsets.device).cumsum(0)
    return new_coords, new_feats, new_labels, new_offsets

def apply_density_downsampling_batch(coords, feats, labels, offsets, prob_apply: float):
    if prob_apply <= 0.0:
        return coords, feats, labels, offsets
    device = feats.device
    dtype_offsets = offsets.dtype
    rng = torch.rand(int(offsets.numel()), device=device)
    apply_mask = rng < float(prob_apply)
    if not torch.any(apply_mask):
        return coords, feats, labels, offsets

    divisors_choices = torch.tensor([2, 3, 4, 5], device=device)
    batch_size = int(offsets.numel())
    start_idx = 0
    new_coords_list = []
    new_feats_list = []
    new_labels_list = []
    new_lengths = []
    for b in range(batch_size):
        end_idx = int(offsets[b].item())
        if apply_mask[b]:
            local_coords = coords[start_idx:end_idx]
            local_feats = feats[start_idx:end_idx]
            local_labels = labels[start_idx:end_idx]
            div = int(divisors_choices[torch.randint(0, 4, (1,), device=device)].item())
            phi = local_feats[:, 0]
            order = torch.argsort(phi)
            kept = order[::div]
            if kept.numel() == 0:
                kept = order[:1]
            new_coords_list.append(local_coords[kept])
            new_feats_list.append(local_feats[kept])
            new_labels_list.append(local_labels[kept])
            new_lengths.append(int(kept.numel()))
        else:
            new_coords_list.append(coords[start_idx:end_idx])
            new_feats_list.append(feats[start_idx:end_idx])
            new_labels_list.append(labels[start_idx:end_idx])
            new_lengths.append(int(end_idx - start_idx))
        start_idx = end_idx

    new_coords = torch.cat(new_coords_list, dim=0).contiguous()
    new_feats = torch.cat(new_feats_list, dim=0).contiguous()
    new_labels = torch.cat(new_labels_list, dim=0).contiguous()
    new_offsets = torch.tensor(new_lengths, dtype=dtype_offsets, device=device).cumsum(0)
    return new_coords, new_feats, new_labels, new_offsets


def apply_noise_adding_batch(coords, feats, offsets, micro_noise_max: float = 0.01, prob_noisify: float = 0.0):
    device = feats.device
    n_points_total = feats.shape[0]
    bottom10_mask = torch.zeros(n_points_total, dtype=torch.bool, device=device)

    start_idx = 0
    for b in range(int(offsets.numel())):
        end_idx = int(offsets[b].item())
        if end_idx <= start_idx:
            start_idx = end_idx
            continue
        local_I = feats[start_idx:end_idx, 3]
        n_local = int(end_idx - start_idx)
        k = max(1, int(n_local * 0.10))
        thr = torch.kthvalue(local_I, k).values
        local_mask = local_I <= thr
        if not torch.any(local_mask):
            min_idx = torch.argmin(local_I)
            local_mask = torch.zeros_like(local_mask)
            local_mask[min_idx] = True
        bottom10_mask[start_idx:end_idx] = local_mask
        start_idx = end_idx

    if micro_noise_max > 0.0 and torch.any(bottom10_mask):
        idx = torch.nonzero(bottom10_mask, as_tuple=False).squeeze(1)
        xy_noise = (torch.rand((idx.numel(), 2), device=device) * 2.0 - 1.0) * float(micro_noise_max)
        coords[idx, 1:3] = torch.clamp(coords[idx, 1:3] + xy_noise, 1e-6, 1 - 1e-6)
        feats[idx, 1:3] = torch.clamp(feats[idx, 1:3] + xy_noise, 1e-6, 1 - 1e-6)

    loss_mask = torch.ones(n_points_total, dtype=torch.bool, device=device)

    if prob_noisify > 0.0 and torch.any(bottom10_mask):
        idx = torch.nonzero(bottom10_mask, as_tuple=False).squeeze(1)
        full_noise_submask = (torch.rand((idx.numel(),), device=device) < float(prob_noisify))
        if torch.any(full_noise_submask):
            chosen = idx[full_noise_submask]
            rand_xy = torch.rand((chosen.numel(), 2), device=device)
            coords[chosen, 1:3] = rand_xy
            feats[chosen, 1:3] = rand_xy
            loss_mask[chosen] = False

    return coords, feats, loss_mask


def train_one_epoch(loader, model, criterion, optimizer, scaler, epoch, writer, args, scheduler=None):
    model.train()
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    last_clip_lower_now = None
    last_angle_range_now = None
    last_density_prob_now = None
    last_noise_prob_now = None
    
    try:
        total_samples_for_pbar = len(loader.dataset)
    except Exception:
        total_samples_for_pbar = None
    pbar = tqdm(total=total_samples_for_pbar, desc=f"Train Epoch {epoch+1}", unit='samples', dynamic_ncols=True)

    non_blocking = (getattr(args, 'device', 'cuda') == 'cuda' and torch.cuda.is_available())
    device = torch.device('cuda' if non_blocking else 'cpu')

    for i, (coords, feats, miller_labels, offsets, crystal_labels, sample_info_list) in enumerate(loader):
        coords = coords.to(device, non_blocking=non_blocking)
        feats = feats.to(device, non_blocking=non_blocking)
        miller_labels = miller_labels.to(device, non_blocking=non_blocking)
        offsets = offsets.to(device, non_blocking=non_blocking)
        crystal_labels = {k: v.to(device, non_blocking=non_blocking) for k, v in crystal_labels.items()}

        if args.dynamic_angle_clip:
            total_iters = getattr(args, 'total_iters', None)
            if total_iters is None:
                total_iters = args.epochs * max(1, len(loader))
                args.total_iters = total_iters
            global_iter = epoch * max(1, len(loader)) + i
            clip_lower_now = compute_clip_lower(global_iter, total_iters, start_lower=1.0, end_lower=float(args.angle_range_low), method=args.angle_clip_schedule)
            last_clip_lower_now = float(clip_lower_now)
            last_angle_range_now = float(clip_lower_now)
            coords, feats, miller_labels, offsets = apply_angle_clipping_batch(coords, feats, miller_labels, offsets, clip_lower_now)

        if getattr(args, 'dynamic_density_clip', False):
            total_iters = getattr(args, 'total_iters', None)
            if total_iters is None:
                total_iters = args.epochs * max(1, len(loader))
                args.total_iters = total_iters
            global_iter = epoch * max(1, len(loader)) + i
            prob_high = float(getattr(args, 'density_clip_prob_high', 0.5))
            density_prob_now = compute_clip_lower(global_iter, total_iters, start_lower=0.0, end_lower=prob_high, method=getattr(args, 'density_clip_schedule', 'linear'))
            last_density_prob_now = float(density_prob_now)
            coords, feats, miller_labels, offsets = apply_density_downsampling_batch(coords, feats, miller_labels, offsets, float(density_prob_now))

        point_loss_mask = None
        if getattr(args, 'dynamic_noise_adding', False):
            total_iters = getattr(args, 'total_iters', None)
            if total_iters is None:
                total_iters = args.epochs * max(1, len(loader))
                args.total_iters = total_iters
            global_iter = epoch * max(1, len(loader)) + i
            prob_high = float(getattr(args, 'noise_adding_prob_high', 0.5))
            noise_prob_now = compute_clip_lower(global_iter, total_iters, start_lower=0.0, end_lower=prob_high, method=getattr(args, 'noise_adding_schedule', 'linear'))
            last_noise_prob_now = float(noise_prob_now)
            coords, feats, point_loss_mask = apply_noise_adding_batch(coords, feats, offsets, micro_noise_max=0.001, prob_noisify=float(noise_prob_now))

        use_amp = (device.type == 'cuda')
        amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with amp_ctx:
            predictions = model(coords, feats, offsets)
            if bool(getattr(args, 'gate_by_sample_acc', True)) and (epoch > int(getattr(args, 'gate_start_epoch', 10))):
                try:
                    thr = float(getattr(args, 'gate_sample_acc_threshold', 0.10))
                except Exception:
                    thr = 0.10
                pred_h_all = torch.argmax(predictions['h'], dim=1)
                pred_k_all = torch.argmax(predictions['k'], dim=1)
                pred_l_all = torch.argmax(predictions['l'], dim=1)
                gating_mask = torch.ones(pred_h_all.shape[0], dtype=torch.bool, device=pred_h_all.device)
                start_idx = 0
                for b in range(int(offsets.numel())):
                    end_idx = int(offsets[b].item())
                    if end_idx > start_idx:
                        correct_all = (
                            (pred_h_all[start_idx:end_idx] == miller_labels[start_idx:end_idx, 0]) &
                            (pred_k_all[start_idx:end_idx] == miller_labels[start_idx:end_idx, 1]) &
                            (pred_l_all[start_idx:end_idx] == miller_labels[start_idx:end_idx, 2])
                        )
                        acc = correct_all.float().mean()
                        if torch.isfinite(acc) and acc.item() < thr:
                            gating_mask[start_idx:end_idx] = False
                    start_idx = end_idx
                if point_loss_mask is not None:
                    point_loss_mask = point_loss_mask & gating_mask
                else:
                    point_loss_mask = gating_mask

            clip_lattice_loss = epoch > 1 
            loss_dict = criterion(predictions, miller_labels, crystal_labels, offsets, point_loss_mask=point_loss_mask, clip_lattice_loss=clip_lattice_loss)
            loss = loss_dict['total_loss']

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            total_steps = getattr(scheduler, 'total_steps', None)
            if total_steps is None or (scheduler.last_epoch + 1) < total_steps:
                scheduler.step()
        
        try:
            batch_num_samples = int(offsets.numel())
        except Exception:
            batch_num_samples = 1
        pbar.update(batch_num_samples)
        pbar.set_postfix({
            'L_miller': f"{loss_dict['loss_miller'].detach().item():.4f}",
            'L_lattice': f"{loss_dict['loss_lattice'].detach().item():.4f}",
            'L_sg': f"{loss_dict['loss_sg'].detach().item():.4f}",
        })

        log_freq = max(1, int(getattr(args, 'log_freq', 100)))
        if (i + 1) % log_freq == 0:
            global_step = epoch * max(1, len(loader)) + i
            log_train_scalars(
                writer,
                loss.detach().item(),
                loss_dict,
                optimizer.param_groups[0]['lr'],
                global_step,
                clip_lower_now=last_clip_lower_now if args.dynamic_angle_clip else None,
                angle_range_now=last_angle_range_now if getattr(args, 'dynamic_angle_clip', False) else None,
                density_clip_prob_now=last_density_prob_now if getattr(args, 'dynamic_density_clip', False) else None,
                noise_adding_prob_now=last_noise_prob_now if getattr(args, 'dynamic_noise_adding', False) else None,
            )

    pbar.close()

def build_eval_loader(paths, miller_index_offset, args, fixed_clip_fraction=None):
    dataset = MillerDataset(
        paths=paths,
        miller_index_offset=miller_index_offset,
        augment_angle=False,
        augment_scale=False,
        debug=0,
        
        norm_scale=args.norm_scale,
        fixed_clip_fraction=fixed_clip_fraction
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

def main():
    args = get_parser().parse_args()
    
    miller_index_offset = -args.min_hkl
    num_classes = args.max_hkl - args.min_hkl + 1
    print("--- Range Info ---")
    print(f"  hkl range: [{args.min_hkl}, {args.max_hkl}]")
    print(f"  Miller Index Offset: {miller_index_offset}")
    print(f"  num_classes: {num_classes}")
    print("--------------------")

    args.save_path = os.path.join(args.save_path, f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_path)
    
    print(f"--> Results will be saved to: {args.save_path}")
    print("--> Args:")
    for arg, value in vars(args).items():
        print(f"    {arg}: {value}")

    train_paths, val_paths = build_paths(args)

    train_dataset, val_dataset = build_datasets(train_paths, val_paths, miller_index_offset, args)
    train_loader = build_train_loader(train_dataset, args)
    
    in_channels = 4

    print(f"--> Input feature dim: {in_channels}, Miller classes: {num_classes}")
    assert in_channels == 4, f"Input feature dim should be 4, got {in_channels}"

    requested = getattr(args, 'device', 'cuda').lower()
    if requested == 'cuda' and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available; falling back to CPU")
        requested = 'cpu'
    device = torch.device('cuda' if (requested == 'cuda' and torch.cuda.is_available()) else 'cpu')

    model = RCT(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"--> Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    
    criterion = CombinedLoss(miller_weight=args.loss_weights[0], lattice_weight=args.loss_weights[1], sg_weight=args.loss_weights[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = create_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"--> Resume from checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
    elif args.pretrained:
        ckpt = load_pretrained_flex(model, args.pretrained)

    try:
        args.total_iters = args.epochs * len(train_loader)
    except Exception:
        args.total_iters = None

    print("--> Start training...")
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, writer, args, scheduler)
        
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        print("--> Fast evaluation...")
        val_metrics = run_fast_eval(val_paths, miller_index_offset, args, model, criterion)

        train_metrics = None
        if (epoch + 1) % args.full_eval_freq == 0:
            print("--> Full evaluation (multi-angle with plots) ...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            full_root = os.path.join(args.save_path, f"full_eval_{timestamp}")
            os.makedirs(full_root, exist_ok=True)
            run_full_eval(val_paths, miller_index_offset, args, model, criterion, full_root)
        
        print("-" * 80)
        print(f"Epoch [{epoch+1}/{args.epochs}] Results:")
        print(f"  VAL SET               | Loss: {val_metrics['loss']:.4f} | L_miller: {val_metrics['loss_miller']:.4f} | L_lattice: {val_metrics['loss_lattice']:.4f} | L_sg: {val_metrics['loss_sg']:.4f} | Miller Acc: {val_metrics['acc_all']:.2f}% | H Acc: {val_metrics['acc_h']:.2f}% | K Acc: {val_metrics['acc_k']:.2f}% | L Acc: {val_metrics['acc_l']:.2f}% | SG Acc: {val_metrics['sg_acc']:.2f}%")
        if train_metrics is not None:
            print(f"  FULL EVAL (VAL multi-angle done) | see: {full_root}")
        print("-" * 80)

        log_val_metrics(writer, val_metrics, epoch)

        is_best = val_metrics['acc_all'] > best_val_acc
        best_val_acc = max(val_metrics['acc_all'], best_val_acc)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_val_acc,
            'args': args
        }, is_best, args.save_path)
    
    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    main()
