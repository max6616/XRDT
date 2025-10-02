import argparse
import os
import time
from datetime import datetime
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from XRDT.model import XRDT
from XRDT.dataset import MillerDataset, collate_fn_offset
from XRDT.loss import CombinedLoss
import warnings
from math import pi, cos
warnings.filterwarnings("ignore")

import multiprocessing as mp
from XRDT.utils import (
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
    parser = argparse.ArgumentParser(description='XRDT')
    parser.add_argument('--model',          type=str,   default='XRDT')
    parser.add_argument('--data_paths',     nargs='+',  default=[
                                                            '/media/max/Data/datasets/mp_random_150k_v1_canonical', 
                                                            '/media/max/Data/datasets/mp_random_150k_v2_canonical', 
                                                            '/media/max/Data/datasets/mp_random_150k_v3_canonical', 
                                                            ])
    parser.add_argument('--save_path',      type=str,   default='/media/max/Data/results/xrd_transformer/v1-3_canonical_ang_clip_density_clip_noise')
    parser.add_argument('--pretrained',     type=str,   default='/media/max/Data/results/xrd_transformer/v1-3_canonical_ang_clip_density_clip/XRDT_20250926_234815/best_model.pth', help='pretrained model path, only load model weights')
    parser.add_argument('--resume',         type=str,   default=None, help='checkpoint path, load full training state')
    parser.add_argument('--debug',          type=int,   default=0, help='if > 0, limit training set size for fast debugging')
    parser.add_argument('--log_freq',       type=int,   default=100, help='tensorboard logging frequency (iterations)')
    parser.add_argument('--full_eval_freq', type=int,   default=40, help='full evaluation frequency (epochs)')
    parser.add_argument('--workers',        type=int,   default=32)
    parser.add_argument('--epochs',         type=int,   default=40)
    parser.add_argument('--batch_size',     type=int,   default=48)
    parser.add_argument('--lr',             type=float, default=5e-5)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--augment_angle',  type=bool,  default=True, help='enable angle augmentation')
    parser.add_argument('--norm_scale',     type=bool,  default=True, help='enable coordinate normalization')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--loss_weights',   type=float, default=[1.0, 5.0, 0.2], nargs=3, help='[L_miller, L_lattice, L_sg]')
    parser.add_argument('--min_hkl',        type=int,   default=-5)
    parser.add_argument('--max_hkl',        type=int,   default=5)
    parser.add_argument('--warmup_epochs',  type=int,   default=1)
    parser.add_argument('--warmup_method',  type=str,   default='cos', choices=['linear', 'cos'])
    # dynamic angle clipping
    parser.add_argument('--dynamic_angle_clip',     type=bool,  default=True, help='enable dynamic angle clipping')
    parser.add_argument('--angle_clip_schedule',    type=str,   default='cos', choices=['linear', 'cos'], help='dynamic angle clipping method')
    parser.add_argument('--angle_range_low',        type=float, default=0.083334, help='angle clipping lower bound')
    # dynamic density clipping (probabilistic angular downsampling per sample)
    parser.add_argument('--dynamic_density_clip',   type=bool,  default=True, help='enable dynamic density downsampling along angle')
    parser.add_argument('--density_clip_schedule',  type=str,   default='cos', choices=['linear', 'cos'], help='schedule for density clip probability growth')
    parser.add_argument('--density_clip_prob_high', type=float, default=0.5, help='upper bound probability to apply density clip to a sample')
    # dynamic noise adding
    parser.add_argument('--dynamic_noise_adding',   type=bool,  default=True, help='enable dynamic noise adding (only process points of lower 10% intensity)')
    parser.add_argument('--noise_adding_schedule',  type=str,   default='cos', choices=['linear', 'cos'], help='schedule for noise adding probability growth')
    parser.add_argument('--noise_adding_prob_high', type=float, default=0.05, help='upper bound probability to apply noise adding to lower 10% points')
    # gradient accumulation
    parser.add_argument('--grad_accum_steps',       type=int,   default=1, help='number of micro-steps to accumulate before optimizer step')
    return parser

def load_pretrained_flex(model: torch.nn.Module, ckpt_path: str):
    print(f"--> Load pretrained weights from '{ckpt_path}' (shape-checked)")
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
    """
    Randomly downsample points along angle dimension per sample with probability prob_apply.
    For each sample chosen, pick a divisor from {2,3,4,5} uniformly and keep every k-th point
    after sorting by angle. Implemented with vectorized indexing per sample for efficiency.
    """
    if prob_apply <= 0.0:
        return coords, feats, labels, offsets
    device = feats.device
    dtype_offsets = offsets.dtype
    rng = torch.rand(int(offsets.numel()), device=device)
    # choose which samples to apply
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
            # select divisor
            div = int(divisors_choices[torch.randint(0, 4, (1,), device=device)].item())
            # sort by angle (phi in feats[:,0])
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
    """
    Add small perturbation to XY for all points while keeping angle unchanged, and
    convert a proportion of points into fully random noise (XY) according to prob_noisify.
    Fully-noised points will be masked out from Miller loss.
    """
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


def train_one_epoch(loader, model, criterion, optimizer, scaler, epoch, writer, args, scheduler=None):
    model.train()
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    last_clip_lower_now = None
    last_angle_range_now = None
    last_density_prob_now = None
    last_noise_prob_now = None
    accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 1)))
    
    # Progress bar configured by number of samples
    try:
        total_samples_for_pbar = len(loader.dataset)
    except Exception:
        total_samples_for_pbar = None
    pbar = tqdm(total=total_samples_for_pbar, desc=f"Train Epoch {epoch+1}", unit='samples', dynamic_ncols=True)

    for i, (coords, feats, miller_labels, offsets, crystal_labels, sample_info_list) in enumerate(loader):
        coords, feats, miller_labels, offsets = coords.cuda(non_blocking=True), feats.cuda(non_blocking=True), miller_labels.cuda(non_blocking=True), offsets.cuda(non_blocking=True)
        crystal_labels = {k: v.cuda(non_blocking=True) for k, v in crystal_labels.items()}

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

        # dynamic density downsampling along angle
        if getattr(args, 'dynamic_density_clip', False):
            total_iters = getattr(args, 'total_iters', None)
            if total_iters is None:
                total_iters = args.epochs * max(1, len(loader))
                args.total_iters = total_iters
            global_iter = epoch * max(1, len(loader)) + i
            # schedule probability from 0 -> density_clip_prob_high
            prob_high = float(getattr(args, 'density_clip_prob_high', 0.5))
            density_prob_now = compute_clip_lower(global_iter, total_iters, start_lower=0.0, end_lower=prob_high, method=getattr(args, 'density_clip_schedule', 'linear'))
            last_density_prob_now = float(density_prob_now)
            coords, feats, miller_labels, offsets = apply_density_downsampling_batch(coords, feats, miller_labels, offsets, float(density_prob_now))

        # dynamic noise adding
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
            coords, feats, point_loss_mask = apply_noise_adding_batch(coords, feats, offsets, micro_noise_max=0.01, prob_noisify=float(noise_prob_now))

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(coords, feats, offsets)
            loss_dict = criterion(predictions, miller_labels, crystal_labels, offsets, point_loss_mask=point_loss_mask)
            loss = loss_dict['total_loss']

        # scale loss for gradient accumulation
        loss_scaled = loss / accum_steps
        scaler.scale(loss_scaled).backward()
        
        do_step = ((i + 1) % accum_steps == 0) or (i + 1 == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        
        # progress bar update
        try:
            batch_num_samples = int(offsets.numel())
        except Exception:
            batch_num_samples = 1
        pbar.update(batch_num_samples)
        # show latest key losses on the bar
        pbar.set_postfix({
            'L_miller': f"{loss_dict['loss_miller'].detach().item():.4f}",
            'L_lattice': f"{loss_dict['loss_lattice'].detach().item():.4f}",
            'L_sg': f"{loss_dict['loss_sg'].detach().item():.4f}",
        })

        # tensorboard logging controlled by log_freq
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
    print(f"--> Args: {args}")

    train_paths, val_paths = build_paths(args)
    # print(f"--> Train set paths: {train_paths}")
    # print(f"--> Val set paths: {val_paths}")

    train_dataset, val_dataset = build_datasets(train_paths, val_paths, miller_index_offset, args)
    train_loader = build_train_loader(train_dataset, args)
    
    in_channels = 4

    print(f"--> Input feature dim: {in_channels}, Miller classes: {num_classes}")
    assert in_channels == 4, f"Input feature dim should be 4, got {in_channels}"

    model = XRDT(in_channels=in_channels, num_classes=num_classes).cuda()
    print(f"--> Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    
    criterion = CombinedLoss(miller_weight=args.loss_weights[0], lattice_weight=args.loss_weights[1], sg_weight=args.loss_weights[2]).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = create_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
    
    scaler = torch.amp.GradScaler(enabled=True)

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

        # Save checkpoint
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
    torch.cuda.empty_cache()
    gc.collect()
    main()
