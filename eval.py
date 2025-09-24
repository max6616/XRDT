import argparse
import os
import time
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from XRDT.model import XRDT
from XRDT.dataset import MillerDataset, collate_fn_offset
from XRDT.loss import CombinedLoss

warnings.filterwarnings("ignore")

from cctbx import crystal, miller
from cctbx.array_family import flex

def get_parser():
    parser = argparse.ArgumentParser(description='High-Performance Point Transformer Evaluation')
    parser.add_argument('--model', type=str, default='pt_v3', help='要使用的模型架构')
    parser.add_argument('--data_paths', nargs='+', default=[
        '/media/max/Data/datasets/mp_random_150k_v1_canonical', 
        # '/media/max/Data/datasets/mp_random_150k_v2_canonical',
        # '/media/max/Data/datasets/mp_random_150k_v3_canonical'
        ], help='多个数据集路径，用空格分隔')
    parser.add_argument('--save_path', type=str, default='./eval_results', help='保存评估结果的路径')
    parser.add_argument('--checkpoint', type=str, default='pretrained/best_model_v123_angle_limited.pth', help='要评估的模型checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--workers', type=int, default=24, help='数据加载的工作进程数')
    parser.add_argument('--min_hkl', type=int, default=-5, help='数据集中hkl的全局最小值 (由分析脚本得到)')
    parser.add_argument('--max_hkl', type=int, default=5, help='数据集中hkl的全局最大值 (由分析脚本得到)')
    # 单标签评估模式，无需多假设参数
    parser.add_argument('--abs_label', type=bool, default=False, help='是否使用绝对值')
    parser.add_argument('--norm_scale', default=True, help='启用坐标缩放归一化')
    parser.add_argument('--debug', type=int, default=0, help='调试模式, 0表示不使用')
    parser.add_argument('--eval_set', type=str, default='val', choices=['train', 'val', 'test'], help='要评估的数据集')
    # 角度范围评估相关
    parser.add_argument('--angle_eval_ranges', nargs='+', type=float, default=[1.0, 0.5, 0.25, 0.1667], help='按比例的角度裁切范围 (1.0=360°, 0.5=180°, 0.1667≈60°, 0.0278≈10°)')
    # HKL canonicalization control
    parser.add_argument('--canonicalize_hkl', type=str, default='gt', choices=['none', 'gt', 'pred'], help='是否对预测HKL进行ASU canonicalize，使用gt或pred空间群')
    return parser

def evaluate(loader, model, criterion, save_path=None, eval_set_name='val', canonicalize_hkl='none', miller_index_offset=0):
    model.eval()
    total_loss, h_correct, k_correct, l_correct, all_correct, total_points = 0, 0, 0, 0, 0, 0
    hkl_loss, lattice_loss, sg_loss = 0, 0, 0
    total_lattice_mae_a, total_lattice_mae_ang = 0, 0
    sg_correct_total, total_samples_with_crystal_info = 0, 0
    
    # 用于收集每个样本的详细信息
    sample_details = []
    
    # 用于收集晶格参数误差数据
    abc_errors_all = []
    ang_errors_all = []
    
    # 初始化扩展后的统计字典，用于存储所有需要的指标
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
    
    with torch.no_grad():
        for i, (coords, feats, miller_labels, offsets, crystal_labels, sample_info_list) in enumerate(tqdm(loader, desc='Evaluating')):
            coords, feats, miller_labels, offsets = coords.cuda(non_blocking=True), feats.cuda(non_blocking=True), miller_labels.cuda(non_blocking=True), offsets.cuda(non_blocking=True)
            crystal_labels = {k: v.cuda(non_blocking=True) for k, v in crystal_labels.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(coords, feats, offsets)
                loss_dict = criterion(predictions, miller_labels, crystal_labels, offsets)
            
            total_loss += loss_dict['total_loss'].detach().item() * len(offsets)
            hkl_loss += loss_dict['loss_miller'].detach().item() * len(offsets)
            lattice_loss += loss_dict['loss_lattice'].detach().item() * len(offsets)
            sg_loss += loss_dict['loss_sg'].detach().item() * len(offsets)

            # --- 按样本收集hkl准确度（单标签） ---
            pred_h_all = torch.argmax(predictions['h'], dim=1)
            pred_k_all = torch.argmax(predictions['k'], dim=1)
            pred_l_all = torch.argmax(predictions['l'], dim=1)
            
            start_idx = 0
            for sample_idx in range(len(offsets)):
                end_idx = offsets[sample_idx]
                num_points_in_sample = end_idx - start_idx
                
                # 直接使用单标签 [N,3]
                sample_labels_flat = miller_labels[start_idx:end_idx]
                labels_h = sample_labels_flat[:, 0]
                labels_k = sample_labels_flat[:, 1]
                labels_l = sample_labels_flat[:, 2]
                
                # 获取该样本的预测
                pred_h = pred_h_all[start_idx:end_idx]
                pred_k = pred_k_all[start_idx:end_idx]
                pred_l = pred_l_all[start_idx:end_idx]

                # 可选：对预测HKL执行ASU canonicalize
                if canonicalize_hkl != 'none':
                    if canonicalize_hkl == 'gt':
                        sg = crystal_labels['space_group'][sample_idx].item() + 1
                    else:  # 'pred'
                        sg = int(torch.argmax(predictions['space_group'][sample_idx]).item()) + 1

                    # 解码为整数HKL
                    h_int = (pred_h - miller_index_offset).detach().cpu().numpy().astype(int)
                    k_int = (pred_k - miller_index_offset).detach().cpu().numpy().astype(int)
                    l_int = (pred_l - miller_index_offset).detach().cpu().numpy().astype(int)
                    triplets = list(zip(h_int.tolist(), k_int.tolist(), l_int.tolist()))
                    # print(sg)

                    canon_triplets = _canonicalize_hkl_batch_cctbx(triplets, sg)
                    # 重新编码为类别索引
                    canon_h = torch.tensor([t[0] + miller_index_offset for t in canon_triplets], device=pred_h.device, dtype=pred_h.dtype)
                    canon_k = torch.tensor([t[1] + miller_index_offset for t in canon_triplets], device=pred_k.device, dtype=pred_k.dtype)
                    canon_l = torch.tensor([t[2] + miller_index_offset for t in canon_triplets], device=pred_l.device, dtype=pred_l.dtype)

                    pred_h, pred_k, pred_l = canon_h, canon_k, canon_l
                
                # 计算该样本的正确数
                sample_h_correct = (pred_h == labels_h).sum().item()
                sample_k_correct = (pred_k == labels_k).sum().item()
                sample_l_correct = (pred_l == labels_l).sum().item()
                sample_all_correct = ((pred_h == labels_h) & (pred_k == labels_k) & (pred_l == labels_l)).sum().item()
                
                # 计算该样本的准确率
                sample_accuracy = sample_all_correct / num_points_in_sample * 100 if num_points_in_sample > 0 else 0
                
                # 获取样本信息
                sample_info = sample_info_list[sample_idx]
                sg_idx = crystal_labels['space_group'][sample_idx].item()
                
                # 收集样本详细信息
                sample_detail = {
                    'filename': sample_info['filename'],
                    'accuracy': float(sample_accuracy),
                    'space_group': int(sg_idx),
                    'total_points': int(num_points_in_sample),
                    'correct_points': int(sample_all_correct)
                }
                sample_details.append(sample_detail)
                
                # 累加到全局统计
                h_correct += sample_h_correct
                k_correct += sample_k_correct
                l_correct += sample_l_correct
                all_correct += sample_all_correct
                total_points += num_points_in_sample
                
                # 如果有空间群信息，累加到按空间群分类的统计中
                if sg_idx != -1:
                    stats_by_sg[sg_idx]['h_correct'] += sample_h_correct
                    stats_by_sg[sg_idx]['k_correct'] += sample_k_correct
                    stats_by_sg[sg_idx]['l_correct'] += sample_l_correct
                    stats_by_sg[sg_idx]['all_correct'] += sample_all_correct
                    stats_by_sg[sg_idx]['total_points'] += num_points_in_sample
                
                start_idx = end_idx
                
            # --- 收集晶格和空间群准确度 ---
            valid_crystal_mask = (crystal_labels['space_group'] != -1).squeeze()
            num_valid_samples = valid_crystal_mask.sum().item()
            if num_valid_samples > 0:
                pred_lattice = predictions['lattice_params'][valid_crystal_mask]
                target_lattice = crystal_labels['lattice'][valid_crystal_mask]
                pred_lattice_unnorm = pred_lattice.clone(); target_lattice_unnorm = target_lattice.clone()
                pred_lattice_unnorm[:, :3] *= 10; target_lattice_unnorm[:, :3] *= 10
                pred_lattice_unnorm[:, 3:] *= 180; target_lattice_unnorm[:, 3:] *= 180
                
                # 计算abc和角度的绝对误差
                abc_errors = torch.abs(pred_lattice_unnorm[:, :3] - target_lattice_unnorm[:, :3])
                ang_errors = torch.abs(pred_lattice_unnorm[:, 3:] - target_lattice_unnorm[:, 3:])
                
                # 累加MAE
                total_lattice_mae_a += torch.nn.functional.l1_loss(pred_lattice_unnorm[:, :3], target_lattice_unnorm[:, :3], reduction='sum').item()
                total_lattice_mae_ang += torch.nn.functional.l1_loss(pred_lattice_unnorm[:, 3:], target_lattice_unnorm[:, 3:], reduction='sum').item()
                
                # 收集所有误差用于累积分布图
                abc_errors_all.extend(abc_errors.flatten().cpu().numpy())
                ang_errors_all.extend(ang_errors.flatten().cpu().numpy())

                pred_sg = torch.argmax(predictions['space_group'][valid_crystal_mask], dim=1)
                target_sg = crystal_labels['space_group'][valid_crystal_mask].squeeze()
                sg_correct_total += (pred_sg == target_sg).sum().item()
                total_samples_with_crystal_info += num_valid_samples

    # --- 绘图部分 ---
    if total_samples_with_crystal_info > 0 and save_path is not None:
        print(f"[{eval_set_name.upper()}] Generating and saving evaluation charts...")
        # 调用封装好的绘图函数，生成4张hkl准确率图
        plot_accuracy_by_sg(stats_by_sg, crystal_systems, save_path, eval_set_name, 'h')
        plot_accuracy_by_sg(stats_by_sg, crystal_systems, save_path, eval_set_name, 'k')
        plot_accuracy_by_sg(stats_by_sg, crystal_systems, save_path, eval_set_name, 'l')
        plot_accuracy_by_sg(stats_by_sg, crystal_systems, save_path, eval_set_name, 'all')
        
        # 生成晶格参数累积分布图
        if len(abc_errors_all) > 0:
            plot_lattice_cumulative_distribution(abc_errors_all, ang_errors_all, save_path, eval_set_name)

    # 保存样本详细信息到JSON文件
    if save_path is not None and sample_details:
        # 按准确率从低到高排序
        sample_details_sorted = sorted(sample_details, key=lambda x: x['accuracy'])

        # 确保eval_results目录存在
        eval_results_path = save_path
        os.makedirs(eval_results_path, exist_ok=True)
        
        # 保存JSON文件
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
    根据提供的统计数据，绘制按空间群分类的准确率点图。

    Args:
        stats_by_sg (dict): 包含按空间群分类统计信息的字典。
        crystal_systems (dict): 晶系到空间群编号范围的映射。
        save_path (str): 保存绘图的根目录。
        eval_set_name (str): 评估集的名称 ('train' 或 'val')。
        acc_type (str): 要绘制的准确率类型 ('h', 'k', 'l', 'all')。
    """
    accuracies = []
    colors = []
    sizes = []
    color_map = {
        'triclinic': '#FF0000', 'monoclinic': '#FF7F00', 'orthorhombic': '#FFFF00',
        'tetragonal': '#00FF00', 'trigonal': '#0000FF', 'hexagonal': '#4B0082', 'cubic': '#9400D3'
    }

    for sg_idx in range(230):
        stats = stats_by_sg.get(sg_idx, {})
        total_points = stats.get('total_points', 0)
        
        if acc_type == 'all':
            correct_points = stats.get('all_correct', 0)
            title = 'Overall HKL Accuracy'
        else: # h, k, or l
            correct_points = stats.get(f'{acc_type}_correct', 0)
            title = f'{acc_type.upper()} Accuracy'

        # 确保数值转换为Python类型，防止tensor导致的绘图错误
        if hasattr(correct_points, 'item'):
            correct_points = correct_points.item()
        if hasattr(total_points, 'item'):
            total_points = total_points.item()
            
        acc = (correct_points / total_points * 100) if total_points > 0 else 0
        accuracies.append(acc)
        
        # 根据数据点数量设置点的大小，最小20，最大100
        point_size = max(10, min(200, total_points / 10)) if total_points > 0 else 20
        sizes.append(point_size)
        
        for system, sg_range in crystal_systems.items():
            if sg_idx + 1 in sg_range:
                colors.append(color_map[system])
                break
    
    plt.figure(figsize=(20, 8))
    
    # 绘制散点图
    x_positions = range(1, 231)
    scatter = plt.scatter(x_positions, accuracies, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Space Group Number', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{title} by Space Group ({eval_set_name.capitalize()} Set)', fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, 231)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 创建图例
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7) for color in color_map.values()]
    plt.legend(legend_elements, color_map.keys(), loc='upper right', title='Crystal Systems')
    
    # 设置x轴刻度
    plt.xticks(range(0, 231, 20))
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存绘图
    filename = f'{eval_set_name}_accuracy_{acc_type}_by_sg.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lattice_cumulative_distribution(abc_errors, ang_errors, save_path, eval_set_name):
    """
    绘制晶格参数的累积分布图。
    
    Args:
        abc_errors (list): abc参数的绝对误差列表
        ang_errors (list): 角度参数的绝对误差列表
        save_path (str): 保存绘图的目录
        eval_set_name (str): 评估集的名称
    """
    # 创建图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制abc参数的累积分布图
    if len(abc_errors) > 0:
        abc_errors = np.array(abc_errors)
        # 定义差值范围：0.01到1
        abc_thresholds = np.logspace(-2, 0, 1000)  # 0.01到1，对数分布
        
        abc_cumulative = []
        for threshold in abc_thresholds:
            cumulative_ratio = np.sum(abc_errors <= threshold) / len(abc_errors) * 100
            abc_cumulative.append(cumulative_ratio)
        
        ax1.plot(abc_thresholds, abc_cumulative, 'b-', linewidth=2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Absolute Error in ABC (Å)', fontsize=12)
        ax1.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax1.set_title(f'Cell Length Accuracy Distribution\n({eval_set_name.capitalize()} Set)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.01, 1.0)
        ax1.set_ylim(0, 100)
        
        # 添加一些关键点的标注
        for threshold in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if threshold in abc_thresholds:
                idx = np.argmin(np.abs(abc_thresholds - threshold))
                ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
                ax1.text(threshold*1.2, abc_cumulative[idx], f'{abc_cumulative[idx]:.1f}%', 
                        fontsize=10, ha='left', va='center')
    
    # 绘制角度参数的累积分布图
    if len(ang_errors) > 0:
        ang_errors = np.array(ang_errors)
        # 定义差值范围：0.1到10度
        ang_thresholds = np.logspace(-1, 1, 1000)  # 0.1到10，对数分布
        
        ang_cumulative = []
        for threshold in ang_thresholds:
            cumulative_ratio = np.sum(ang_errors <= threshold) / len(ang_errors) * 100
            ang_cumulative.append(cumulative_ratio)
        
        ax2.plot(ang_thresholds, ang_cumulative, 'r-', linewidth=2)
        ax2.set_xscale('log')
        ax2.set_xlabel('Absolute Error in Angles (degrees)', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title(f'Cell Angles Accuracy Distribution\n({eval_set_name.capitalize()} Set)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.1, 10.0)
        ax2.set_ylim(0, 100)
        
        # 添加一些关键点的标注
        for threshold in [0.1, 0.5, 1.0, 5.0, 10.0]:
            if threshold in ang_thresholds:
                idx = np.argmin(np.abs(ang_thresholds - threshold))
                ax2.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
                ax2.text(threshold*1.2, ang_cumulative[idx], f'{ang_cumulative[idx]:.1f}%', 
                        fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    
    # 确保保存目录存在
    eval_results_path = save_path
    os.makedirs(eval_results_path, exist_ok=True)
    
    # 保存图形
    filename = f'{eval_set_name}_lattice_cumulative_distribution.png'
    plt.savefig(os.path.join(eval_results_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{eval_set_name.upper()}] Lattice cumulative distribution charts saved to: {filename}")

def _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=None):
    eval_dataset = MillerDataset(
        paths=eval_paths,
        miller_index_offset=miller_index_offset,
        augment_angle=False,
        augment_scale=False,
        debug=args.debug,
        abs_label=args.abs_label,
        norm_scale=args.norm_scale,
        fixed_clip_fraction=fixed_clip_fraction
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
    print("--- 动态范围参数 ---")
    print(f"  hkl 范围: [{args.min_hkl}, {args.max_hkl}]")
    print(f"  计算得到的 Miller Index Offset: {miller_index_offset}")
    print(f"  计算得到的 num_classes: {num_classes}")
    print("--------------------")

    # 创建主保存目录（时间戳）
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    root_save_dir = os.path.join(args.save_path, f"eval_{timestamp}")
    os.makedirs(root_save_dir, exist_ok=True)
    print(f"--> 评估结果将保存在: {root_save_dir}")
    print(f"--> 参数: {args}")

    # 处理数据集路径
    if args.data_paths is not None:
        # 使用多个数据集路径
        eval_paths = [os.path.join(path, args.eval_set) for path in args.data_paths]
        print(f"--> 评估数据集路径: {eval_paths}")
    else:
        print("错误: 必须指定数据集路径")
        return

    # 首先构建完整数据加载器以推断输入通道
    _, eval_loader_full = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=None)
    
    try:
        _, feats, _, _, _, _ = next(iter(eval_loader_full))
        in_channels = feats.shape[1]
    except StopIteration:
        print("无法从dataloader获取数据，将使用默认输入通道4。")
        in_channels = 4

    print(f"--> 检测到输入特征维度: {in_channels}, Miller指数类别数: {num_classes}")
    assert in_channels == 4, f"输入特征维度应为4, 但检测到 {in_channels}"

    # 创建模型
    model = XRDT(in_channels=in_channels, num_classes=num_classes).cuda()
    print(f"--> 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    
    # 加载checkpoint
    print(f"--> 加载checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print("--> 模型参数加载成功")
    
    # 创建损失函数
    criterion = CombinedLoss(miller_weight=1.0, lattice_weight=5.0, sg_weight=0.2).cuda()
    
    # 循环不同角度范围评估
    angle_ranges = args.angle_eval_ranges
    summary_results = {}
    angle_name_map = {}
    for frac in angle_ranges:
        name = f'{int(frac*360)}deg'

        angle_name_map[frac] = name
        subdir = os.path.join(root_save_dir, name)
        os.makedirs(subdir, exist_ok=True)

        # 针对该角度构建数据加载器
        _, eval_loader = _build_dataset_and_loader(eval_paths, miller_index_offset, args, fixed_clip_fraction=frac)

        print(f"--> 开始评估 {args.eval_set} 集，角度范围: {name} ({frac*360:.2f}°)...")
        eval_metrics, eval_extras = evaluate(
            eval_loader,
            model,
            criterion,
            subdir,
            eval_set_name=args.eval_set,
            canonicalize_hkl=args.canonicalize_hkl,
            miller_index_offset=miller_index_offset,
        )

        # 打印评估结果
        print("-" * 80)
        print(f"评估结果 ({args.eval_set.upper()} 集, {name}):")
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

        # 保存到子目录 JSON
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
        # 额外保存便于叠加绘图的数据
        with open(os.path.join(subdir, f'{args.eval_set}_extras.npy'), 'wb') as f:
            np.save(f, {
                'abc_errors': np.array(eval_extras['abc_errors'], dtype=np.float32),
                'ang_errors': np.array(eval_extras['ang_errors'], dtype=np.float32),
            }, allow_pickle=True)
        # 保存stats_by_sg为json（数值化）
        stats_json = {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in eval_extras['stats_by_sg'].items()}
        with open(os.path.join(subdir, f'{args.eval_set}_stats_by_sg.json'), 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, indent=2, ensure_ascii=False)

        # 清理数据加载器
        del eval_loader
        torch.cuda.empty_cache()

    # 写总汇总到主目录
    with open(os.path.join(root_save_dir, f'summary_{args.eval_set}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    # 叠加绘图：lattice累计分布 & h/k/l准确率
    try:
        overlay_lattice_path = os.path.join(root_save_dir, f'{args.eval_set}_lattice_cumulative_distribution.png')
        overlay_acc_paths = {
            'h': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_h_by_sg.png'),
            'k': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_k_by_sg.png'),
            'l': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_l_by_sg.png'),
            'all': os.path.join(root_save_dir, f'{args.eval_set}_accuracy_all_by_sg.png'),
        }

        # 加载各子目录数据
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

        # 绘制叠加的lattice累计分布
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

        # 叠加的h/k/l/all点图
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
                # 根据数据点数量设置点的大小
                point_size = max(20, min(100, total_points / 10)) if total_points > 0 else 20
                sizes.append(point_size)
            return np.array(series, dtype=np.float32), np.array(sizes, dtype=np.float32)

        # 为不同角度范围分配稳定颜色
        angle_names = [name for name, _, __, ___ in ordered_items]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {angle_names[i]: color_cycle[i % len(color_cycle)] for i in range(len(angle_names))}

        for key in ['h', 'k', 'l', 'all']:
            plt.figure(figsize=(20, 8))
            x = np.arange(1, 231)

            # 先构建每个角度的series和sizes
            series_by_angle = {}
            sizes_by_angle = {}
            for name, _, __, stats_by_sg in ordered_items:
                series_by_angle[name], sizes_by_angle[name] = build_series(stats_by_sg, key)

            # 绘制每个角度范围的点图
            for name in angle_names:
                series = series_by_angle[name]
                sizes = sizes_by_angle[name]
                # 只绘制有数据的点
                mask = series > 0
                if np.any(mask):
                    plt.scatter(x[mask], series[mask], s=sizes[mask], 
                              c=color_map[name], alpha=0.7, edgecolors='black', 
                              linewidth=0.5, label=name)

            plt.xlabel('Space Group Number', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'{key.upper()} Accuracy by Space Group (Overlay Points)', fontsize=14)
            plt.ylim(0, 100)
            plt.xlim(0, 231)
            
            # 添加网格
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 设置x轴刻度
            tick_positions = np.arange(0, 231, 20)
            plt.xticks(tick_positions)

            # 构建图例
            plt.legend(loc='upper right', title='Angle Ranges')
            plt.tight_layout()
            plt.savefig(overlay_acc_paths[key], dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"生成叠加图失败: {e}")

    print("评估完成.")

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
