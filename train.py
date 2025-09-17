import argparse
import os
import time
from datetime import datetime
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from XRDT.model import MillerIndexerV3
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
    parser.add_argument('--save_path',      type=str,   default='/media/max/Data/results/xrd_transformer/mp_random_150k_canonical')
    parser.add_argument('--pretrained',     type=str,   default=None, help='pretrained model path, only load model weights')
    parser.add_argument('--resume',         type=str,   default=None, help='checkpoint path, load full training state')
    parser.add_argument('--debug',          type=int,   default=0, help='if >0, limit training set size for fast debugging')
    parser.add_argument('--print_freq',     type=int,   default=100, help='print training info frequency (iterations)')
    parser.add_argument('--full_eval_freq', type=int,   default=10, help='full evaluation frequency (epochs)')
    parser.add_argument('--workers',        type=int,   default=24)
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=12)
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
    parser.add_argument('--angle_range_low',        type=float, default=0.0278, help='angle clipping lower bound')
    return parser

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


def train_one_epoch(loader, model, criterion, optimizer, scaler, epoch, writer, args, scheduler=None):
    model.train()
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    last_clip_lower_now = None
    
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
            coords, feats, miller_labels, offsets = apply_angle_clipping_batch(coords, feats, miller_labels, offsets, clip_lower_now)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(coords, feats, offsets)
            loss_dict = criterion(predictions, miller_labels, crystal_labels, offsets)
            loss = loss_dict['total_loss']

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        if (i + 1) % args.print_freq == 0:
            elapsed = time.time() - start_time
            extra_clip = ''
            if args.dynamic_angle_clip and last_clip_lower_now is not None:
                extra_clip = f" | ClipLower: {last_clip_lower_now:.4f}"
            print(f"Epoch: [{epoch+1}/{args.epochs}][{i+1}/{len(loader)}] | "
                f"Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB | "
                f"Loss: {loss.detach().item():.4f} | "
                f"L_miller: {loss_dict['loss_miller'].detach().item():.4f} | "
                f"L_h: {loss_dict['loss_h'].detach().item():.4f} | "
                f"L_k: {loss_dict['loss_k'].detach().item():.4f} | "
                f"L_l: {loss_dict['loss_l'].detach().item():.4f} | "
                f"L_lattice: {loss_dict['loss_lattice'].detach().item():.4f} | "
                f"L_sg: {loss_dict['loss_sg'].detach().item():.4f}{extra_clip} | "
                f"time: {elapsed:.2f}s")
            global_step = epoch * len(loader) + i
            log_train_scalars(
                writer,
                loss.detach().item(),
                loss_dict,
                optimizer.param_groups[0]['lr'],
                global_step,
                clip_lower_now=last_clip_lower_now if args.dynamic_angle_clip else None,
            )
            start_time = time.time()

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

    model = MillerIndexerV3(in_channels=in_channels, num_classes=num_classes).cuda()
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
        print(f"--> Load pretrained weights from '{args.pretrained}'")
        ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    try:
        args.total_iters = args.epochs * len(train_loader)
    except Exception:
        args.total_iters = None

    print("--> Start training...")
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, writer, args, scheduler)
        
        if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            pass
        else:
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
