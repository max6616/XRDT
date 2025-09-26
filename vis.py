import os
import json
from collections import OrderedDict
import argparse

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from pathlib import Path
from cctbx import crystal, miller
from cctbx.array_family import flex
import csv
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Visualize PointTransformerV3 intermediate outputs and predictions.")
    parser.add_argument("--model", default="pretrained/best_model_v123_angle_limited.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", default="/media/max/Data/datasets/mp_random_150k_v3_canonical/test/test_000200.jsonl", help="Path to a jsonl sample file")
    parser.add_argument("--num-classes", type=int, default=11, help="Number of classes")
    parser.add_argument("--in-channels", type=int, default=4, help="Number of input channels")
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda, cuda:0, cpu). Default: auto")
    parser.add_argument("--outdir", default="vis_mask_outputs", help="Directory to save HTML visualizations")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument(
        "--canonicalize-hkl",
        default="gt",
        choices=["none", "gt", "pred"],
        help="Apply cctbx ASU mapping to predictions using gt or predicted space group",
    )
    args = parser.parse_args()


    from XRDT.model import XRDT

    model_path = args.model
    data_file = args.data
    num_classes = args.num_classes
    in_channels = args.in_channels
    # masking removed

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"--> Using device: {device}")

    print(f"--> Loading model: {model_path}")
    model = XRDT(in_channels=in_channels, num_classes=num_classes)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    try:
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    except Exception as exc:
        print(f"--> Strict load failed: {exc}")
        print("--> Falling back to non-strict load")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    print("--> Model loaded.")

    print(f"--> Reading data from: {data_file}")
    with open(data_file, "r") as f:
        line = f.readline()
        sample_data = json.loads(line)

    coords_feats = torch.tensor(sample_data["input_sequence"], dtype=torch.float32)
    labels_raw = torch.tensor(sample_data["labels"], dtype=torch.long)
    if labels_raw.ndim == 3:
        labels_raw = labels_raw[0]

    coords_only = coords_feats[:, :3]
    is_abs = num_classes < 11
    miller_offset = 0 if is_abs else 5

    if is_abs:
        hkl_features = torch.abs(labels_raw).clone().float() / 5.0
    else:
        hkl_features = labels_raw.clone().float() / 5.0

    # Intensity normalization and minor coordinate normalization per notebook
    coords_feats[:, 3] /= 10.0
    coords_feats[:, 1:3] = (
        (coords_feats[:, 1:3] - 0.5) * 0.99 / torch.max(coords_feats[:, 1:3]) + 0.5
    )

    if in_channels == 4:
        feats_with_hkl = coords_feats
    else:
        feats_with_hkl = torch.cat([coords_feats, hkl_features], dim=1)

    if is_abs:
        labels_final = torch.abs(labels_raw)
    else:
        labels_final = labels_raw + miller_offset

    offsets = torch.tensor([len(coords_only)], dtype=torch.long)
    original_labels_tensor = labels_final
    original_intensities = coords_feats[:, 3] * 10.0

    # Parse GT space group if available
    gt_space_group: int | None = None
    try:
        if isinstance(sample_data, dict):
            if "space_group" in sample_data and sample_data["space_group"] is not None:
                gt_space_group = int(sample_data["space_group"]) if int(sample_data["space_group"]) != -1 else None
            elif "crystal" in sample_data and isinstance(sample_data["crystal"], dict):
                sg_candidate = sample_data["crystal"].get("space_group", None)
                if sg_candidate is not None and int(sg_candidate) != -1:
                    gt_space_group = int(sg_candidate)
            elif "crystal_labels" in sample_data and isinstance(sample_data["crystal_labels"], dict):
                sg_candidate = sample_data["crystal_labels"].get("space_group", None)
                if sg_candidate is not None and int(sg_candidate) != -1:
                    gt_space_group = int(sg_candidate)
    except Exception:
        gt_space_group = None

    print("--> Preprocessing done.")
    print(f"    coords shape: {coords_only.shape}")
    print(f"    feats  shape: {feats_with_hkl.shape}")

    model.to(device)
    coords_only_dev = coords_only.to(device)
    feats_with_hkl_dev = feats_with_hkl.to(device)
    offsets_dev = offsets.to(device)
    original_labels_dev = original_labels_tensor.to(device)
    original_intensities_dev = original_intensities.to(device)

    print("--> Running forward with hooks to capture data ...")
    vis_data = run_and_capture_with_hooks(
        model,
        coords_only_dev,
        feats_with_hkl_dev,
        offsets_dev,
        original_labels_dev,
        original_intensities_dev,
        is_abs_label=is_abs,
        canonicalize_hkl=args.canonicalize_hkl,
        gt_space_group=gt_space_group,
    )
    print("--> Capture done.")

    # Report metrics on all points (raw vs canonicalized)
    try:
        labels_np = vis_data["labels"]
        preds_raw_np = vis_data.get("predictions_raw", None)
        preds_canon_np = vis_data.get("predictions_canon", None)
        num_points_total = len(labels_np)
        print("\n--- Metrics on all points ---")
        print(f"Total points: {num_points_total}")
        if preds_raw_np is not None and num_points_total > 0:
            h_acc = (preds_raw_np[:, 0] == labels_np[:, 0]).mean() * 100.0
            k_acc = (preds_raw_np[:, 1] == labels_np[:, 1]).mean() * 100.0
            l_acc = (preds_raw_np[:, 2] == labels_np[:, 2]).mean() * 100.0
            all_acc = (np.all(preds_raw_np == labels_np, axis=1)).mean() * 100.0
            print(f"Raw   - H: {h_acc:.2f}%  K: {k_acc:.2f}%  L: {l_acc:.2f}%  All: {all_acc:.2f}%")
        if preds_canon_np is not None and num_points_total > 0:
            h_acc_c = (preds_canon_np[:, 0] == labels_np[:, 0]).mean() * 100.0
            k_acc_c = (preds_canon_np[:, 1] == labels_np[:, 1]).mean() * 100.0
            l_acc_c = (preds_canon_np[:, 2] == labels_np[:, 2]).mean() * 100.0
            all_acc_c = (np.all(preds_canon_np == labels_np, axis=1)).mean() * 100.0
            print(f"Canon - H: {h_acc_c:.2f}%  K: {k_acc_c:.2f}%  L: {l_acc_c:.2f}%  All: {all_acc_c:.2f}%")
        print("-" * 40)
    except Exception as _metrics_exc:
        print(f"Metrics computation failed: {_metrics_exc}")

    # Save per-point CSV and metrics JSONL
    try:
        outdir = args.outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)
        base_stem = Path(data_file).stem

        # CSV: points with coords, intensity, label hkl, predicted hkl (raw and canonicalized if available)
        csv_path = os.path.join(outdir, f"{base_stem}_points.csv")
        points_np = vis_data["p0_original"]
        intensities_np = vis_data["intensities"]
        labels_np = vis_data["labels"]
        preds_raw_np = vis_data.get("predictions_raw", None)
        preds_canon_np = vis_data.get("predictions_canon", None)

        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            header = [
                "x",
                "y",
                "z",
                "intensity",
                "label_h",
                "label_k",
                "label_l",
                "pred_h",
                "pred_k",
                "pred_l",
                "pred_h_canon",
                "pred_k_canon",
                "pred_l_canon",
            ]
            writer.writerow(header)
            n = points_np.shape[0]
            for i in range(n):
                pr_h = preds_raw_np[i, 0] if preds_raw_np is not None else ""
                pr_k = preds_raw_np[i, 1] if preds_raw_np is not None else ""
                pr_l = preds_raw_np[i, 2] if preds_raw_np is not None else ""
                pc_h = preds_canon_np[i, 0] if preds_canon_np is not None else ""
                pc_k = preds_canon_np[i, 1] if preds_canon_np is not None else ""
                pc_l = preds_canon_np[i, 2] if preds_canon_np is not None else ""
                writer.writerow(
                    [
                        float(points_np[i, 0]),
                        float(points_np[i, 1]),
                        float(points_np[i, 2]),
                        float(intensities_np[i]),
                        int(labels_np[i, 0]),
                        int(labels_np[i, 1]),
                        int(labels_np[i, 2]),
                        pr_h if pr_h == "" else int(pr_h),
                        pr_k if pr_k == "" else int(pr_k),
                        pr_l if pr_l == "" else int(pr_l),
                        pc_h if pc_h == "" else int(pc_h),
                        pc_k if pc_k == "" else int(pc_k),
                        pc_l if pc_l == "" else int(pc_l),
                    ]
                )
        print(f"Saved: {os.path.abspath(csv_path)}")

        # JSONL: metrics
        jsonl_path = os.path.join(outdir, f"{base_stem}_metrics.jsonl")
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_path": os.path.abspath(model_path),
            "data_path": os.path.abspath(data_file),
            "num_points": int(points_np.shape[0]),
            "canon_mode": vis_data.get("canon_mode"),
            "canon_sg_used": vis_data.get("canon_sg_used"),
            "space_group_gt": vis_data.get("sg_gt"),
            "space_group_pred": vis_data.get("sg_pred"),
        }

        # accuracies (as fractions 0-1)
        try:
            if preds_raw_np is not None and points_np.shape[0] > 0:
                metrics.update(
                    {
                        "h_accuracy_raw": float((preds_raw_np[:, 0] == labels_np[:, 0]).mean()),
                        "k_accuracy_raw": float((preds_raw_np[:, 1] == labels_np[:, 1]).mean()),
                        "l_accuracy_raw": float((preds_raw_np[:, 2] == labels_np[:, 2]).mean()),
                        "indexing_accuracy_raw": float((np.all(preds_raw_np == labels_np, axis=1)).mean()),
                    }
                )
            if preds_canon_np is not None and points_np.shape[0] > 0:
                metrics.update(
                    {
                        "h_accuracy_canon": float((preds_canon_np[:, 0] == labels_np[:, 0]).mean()),
                        "k_accuracy_canon": float((preds_canon_np[:, 1] == labels_np[:, 1]).mean()),
                        "l_accuracy_canon": float((preds_canon_np[:, 2] == labels_np[:, 2]).mean()),
                        "indexing_accuracy_canon": float((np.all(preds_canon_np == labels_np, axis=1)).mean()),
                    }
                )
        except Exception:
            pass

        # space group accuracy if available
        sg_gt = vis_data.get("sg_gt")
        sg_pred = vis_data.get("sg_pred")
        if sg_gt is not None and sg_pred is not None:
            metrics["space_group_accuracy"] = float(1.0 if int(sg_gt) == int(sg_pred) else 0.0)

        with open(jsonl_path, "w") as f_jsl:
            f_jsl.write(json.dumps(metrics) + "\n")
        print(f"Saved: {os.path.abspath(jsonl_path)}")
    except Exception as _save_exc:
        print(f"Saving CSV/JSONL failed: {_save_exc}")

    # Figure 1: Main interactive visualization
    fig = go.Figure()
    add_main_figure(fig, vis_data, is_abs_label=is_abs)
    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    open_in_browser = not args.no_open
    main_path = os.path.join(outdir, "main.html")
    fig.write_html(main_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {os.path.abspath(main_path)}")
    if open_in_browser:
        try:
            webbrowser.open(f"file://{os.path.abspath(main_path)}")
        except Exception as e:
            print(f"Failed to open browser: {e}")

    # mask distribution visualization removed

    # Figure 3: HKL difference visualization
    preds_raw = vis_data.get("predictions_raw", None)
    preds_canon = vis_data.get("predictions_canon", None)
    labels = vis_data["labels"]
    points = vis_data["p0_original"]

    # L1 difference (sum of abs)
    fig_diff = go.Figure()
    buttons = []
    visibility = []
    trace_idx = 0

    if preds_raw is not None:
        diff_raw = np.sum(np.abs(preds_raw - labels), axis=1)
        fig_diff.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=diff_raw,
                    colorscale="Reds",
                    cmin=0,
                    cmax=max(1e-6, diff_raw.max()),
                    colorbar=dict(title="HKL L1 Difference"),
                    line=dict(width=0),
                ),
                customdata=np.hstack((preds_raw, labels, diff_raw[:, np.newaxis])),
                hovertemplate=(
                    "<b>Pred (Raw):</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                    "<b>Label:</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]})<br>"
                    "<b>L1:</b> %{customdata[6]:.2f}<br>"
                    "<b>Coord:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
                ),
                name="Raw",
                visible=True,
            )
        )
        visibility.append(True)
        trace_idx += 1

    if preds_canon is not None:
        diff_canon = np.sum(np.abs(preds_canon - labels), axis=1)
        fig_diff.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=diff_canon,
                    colorscale="Reds",
                    cmin=0,
                    cmax=max(1e-6, diff_canon.max()),
                    colorbar=dict(title="HKL L1 Difference"),
                    line=dict(width=0),
                ),
                customdata=np.hstack((preds_canon, labels, diff_canon[:, np.newaxis])),
                hovertemplate=(
                    "<b>Pred (Canon):</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                    "<b>Label:</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]})<br>"
                    "<b>L1:</b> %{customdata[6]:.2f}<br>"
                    "<b>Coord:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
                ),
                name="Canonicalized",
                visible=False if preds_raw is not None else True,
            )
        )
        visibility.append(False if preds_raw is not None else True)

    buttons.append(
        dict(
            label="Raw",
            method="update",
            args=[{"visible": [True, False] if preds_canon is not None else [True]}, {"title": "HKL difference (Raw)"}],
        )
    )
    if preds_canon is not None:
        buttons.append(
            dict(
                label="Canonicalized",
                method="update",
                args=[{"visible": [False, True]}, {"title": "HKL difference (Canonicalized)"}],
            )
        )

    fig_diff.update_layout(
        title="HKL difference",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="right",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        scene=hidden_axes_scene("data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    diff_path = os.path.join(outdir, "hkl_difference.html")
    fig_diff.write_html(diff_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {os.path.abspath(diff_path)}")
    if open_in_browser:
        try:
            webbrowser.open(f"file://{os.path.abspath(diff_path)}")
        except Exception as e:
            print(f"Failed to open browser: {e}")


def collate_fn_offset(batch):
    """Example collate_fn for batched point data with offsets."""
    coords, feats, labels = [], [], []
    for i, (points, label) in enumerate(batch):
        coords.append(torch.cat([torch.full((points.shape[0], 1), i), points], dim=1))
        feats.append(points[:, :])
        labels.append(label)
    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    p_out = coords[:, 1:]
    offsets = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long).cumsum(0)
    return p_out, feats, labels, offsets


def hkl_to_rgb(hkl_array: np.ndarray, is_abs: bool = False):
    arr = np.abs(hkl_array) if is_abs else hkl_array
    max_val = 5.0
    offset = np.min(arr) * -1
    normalized = (arr + offset) / (max_val + offset + 1e-6)
    rgb_array = (np.clip(normalized, 0, 1) * 230).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb_array]


def intensity_to_size(intensities: np.ndarray):
    min_val, max_val = intensities.min(), intensities.max()
    if max_val == min_val:
        return np.full_like(intensities, 4)
    normalized = (intensities - min_val) / (max_val - min_val)
    return 0 + (normalized + 0.8) ** 4


def hidden_axes_scene(aspectmode: str = "data") -> dict:
    return dict(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showspikes=False,
            showbackground=False,
            ticks="",
            title="",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showspikes=False,
            showbackground=False,
            ticks="",
            title="",
        ),
        zaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showspikes=False,
            showbackground=False,
            ticks="",
            title="",
        ),
        aspectmode=aspectmode,
    )


def _canonicalize_hkl_batch_cctbx(hkl_triplets, sg_number_1_based: int):
    """Canonicalize a list of HKL triplets to the ASU using cctbx.

    hkl_triplets: List[Tuple[int, int, int]]
    sg_number_1_based: int in [1, 230]
    returns List[List[int]] of same length
    """
    symm = crystal.symmetry(space_group_symbol=str(int(sg_number_1_based)))
    ms = miller.set(
        crystal_symmetry=symm,
        indices=flex.miller_index(hkl_triplets),
        anomalous_flag=False,
    )
    ms_asu = ms.map_to_asu()
    idx = ms_asu.indices()
    return [list(idx[i]) for i in range(len(hkl_triplets))]


def run_and_capture_with_hooks(
    model: torch.nn.Module,
    coords_xyz: torch.Tensor,
    features: torch.Tensor,
    offsets: torch.Tensor,
    original_labels: torch.Tensor,
    original_intensities: torch.Tensor,
    is_abs_label: bool,
    canonicalize_hkl: str,
    gt_space_group: int | None,
):
    """Run forward pass with forward hooks and capture coordinates and predictions."""
    captured_coords: dict[str, np.ndarray] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):  # type: ignore[no-untyped-def]
            # Each module is expected to output Point-like structure with .coord
            captured_coords[name] = output.coord.detach().cpu().numpy()

        return hook

    with torch.no_grad():
        modules_to_hook = OrderedDict(
            [
                ("p0_embedding", model.backbone.embedding),
                ("p1_enc", model.backbone.enc.enc0),
                ("p2_enc", model.backbone.enc.enc1),
                ("p3_enc", model.backbone.enc.enc2),
                ("p4_enc", model.backbone.enc.enc3),
                ("p5_enc", model.backbone.enc.enc4),
                ("p4_dec", model.backbone.dec.dec3),
                ("p3_dec", model.backbone.dec.dec2),
                ("p2_dec", model.backbone.dec.dec1),
                ("p1_dec", model.backbone.dec.dec0),
            ]
        )

        for name, module in modules_to_hook.items():
            hooks.append(module.register_forward_hook(make_hook(name)))

        predictions_dict = model(coords_xyz, features, offsets)

        for h in hooks:
            h.remove()

        vis_data = {}
        vis_data.update(captured_coords)
        vis_data["p0_original"] = coords_xyz.detach().cpu().numpy()

        pred_h = torch.argmax(predictions_dict["h"], dim=1)
        pred_k = torch.argmax(predictions_dict["k"], dim=1)
        pred_l = torch.argmax(predictions_dict["l"], dim=1)
        predictions = torch.stack([pred_h, pred_k, pred_l], dim=1)

        miller_offset_val = 0 if is_abs_label else 5

        # Decode to human-readable HKL integers
        preds_raw_hkl = (predictions.detach().cpu().numpy() - miller_offset_val).astype(np.int32)
        labels_hkl = (original_labels.detach().cpu().numpy() - miller_offset_val).astype(np.int32)

        # Predict SG (1-based) from model outputs, single sample assumed
        pred_sg = None
        if "space_group" in predictions_dict:
            try:
                sg_logits = predictions_dict["space_group"]
                if sg_logits.ndim == 2:
                    pred_sg = int(torch.argmax(sg_logits[0]).item()) + 1
                else:
                    pred_sg = int(torch.argmax(sg_logits).item()) + 1
            except Exception:
                pred_sg = None

        # Determine SG used for canonicalization, if needed
        canon_sg_used = None
        if canonicalize_hkl != "none":
            if canonicalize_hkl == "gt" and gt_space_group is not None and int(gt_space_group) != -1:
                canon_sg_used = int(gt_space_group)
            else:
                canon_sg_used = pred_sg

        preds_canon_hkl = None
        if canon_sg_used is not None:
            triplets = [tuple(map(int, t)) for t in preds_raw_hkl.tolist()]
            try:
                canon_triplets = _canonicalize_hkl_batch_cctbx(triplets, canon_sg_used)
                preds_canon_hkl = np.array(canon_triplets, dtype=np.int32)
            except Exception:
                preds_canon_hkl = None

        vis_data["predictions_raw"] = preds_raw_hkl
        if preds_canon_hkl is not None:
            vis_data["predictions_canon"] = preds_canon_hkl
        vis_data["labels"] = labels_hkl
        vis_data["sg_pred"] = pred_sg
        vis_data["sg_gt"] = int(gt_space_group) if gt_space_group is not None else None
        vis_data["canon_mode"] = canonicalize_hkl
        vis_data["canon_sg_used"] = canon_sg_used
        vis_data["intensities"] = original_intensities.detach().cpu().numpy()

    return vis_data


def add_main_figure(fig: go.Figure, vis_data: dict, is_abs_label: bool):
    stages = OrderedDict(
        [
            ("gt_color", {"name": "Ground Truth (color)", "type": "color"}),
            ("pred_color_raw", {"name": "Prediction Raw (color)", "type": "color"}),
            ("pred_color_canon", {"name": "Prediction Canon (color)", "type": "color"}),
            ("gt_flow", {"name": "Ground Truth (direction)", "type": "flow"}),
            ("pred_flow_raw", {"name": "Prediction Raw (direction)", "type": "flow"}),
            ("pred_flow_canon", {"name": "Prediction Canon (direction)", "type": "flow"}),
            ("p0_original", {"name": "Original Points", "type": "structure", "color": "grey"}),
            ("p0_embedding", {"name": "Embedding Out", "type": "structure", "color": "cyan"}),
            ("p1_enc", {"name": "Encoder 1", "type": "structure", "color": "darkorange"}),
            ("p2_enc", {"name": "Encoder 2", "type": "structure", "color": "green"}),
            ("p3_enc", {"name": "Encoder 3", "type": "structure", "color": "firebrick"}),
            ("p4_enc", {"name": "Encoder 4", "type": "structure", "color": "purple"}),
            ("p5_enc", {"name": "Encoder 5 (Bottleneck)", "type": "structure", "color": "saddlebrown"}),
            ("p4_dec", {"name": "Decoder 4", "type": "structure", "color": "mediumpurple"}),
            ("p3_dec", {"name": "Decoder 3", "type": "structure", "color": "lightcoral"}),
            ("p2_dec", {"name": "Decoder 2", "type": "structure", "color": "lightgreen"}),
            ("p1_dec", {"name": "Decoder 1 (Final)", "type": "structure", "color": "sandybrown"}),
        ]
    )

    traces: list[go.Scatter3d] = []
    for key, info in stages.items():
        visible = key == "gt_color"
        if info["type"] == "color":
            if key == "gt_color":
                hkl_data = vis_data["labels"]
            elif key == "pred_color_canon" and "predictions_canon" in vis_data:
                hkl_data = vis_data["predictions_canon"]
            else:
                hkl_data = vis_data.get("predictions_raw", vis_data["labels"])
            points = vis_data["p0_original"]
            intns = vis_data["intensities"]
            # Build customdata: pred_raw(3), pred_canon(3 or NaN), label(3), intensity(1)
            pred_raw = vis_data.get("predictions_raw", None)
            pred_canon = vis_data.get("predictions_canon", None)
            if pred_raw is None:
                pred_raw = np.full_like(vis_data["labels"], np.nan)
            if pred_canon is None:
                pred_canon = np.full_like(vis_data["labels"], np.nan)
            custom = np.hstack((
                pred_raw,
                pred_canon,
                vis_data["labels"],
                intns[:, np.newaxis],
            ))
            traces.append(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=intensity_to_size(intns),
                        color=hkl_to_rgb(hkl_data, is_abs=is_abs_label),
                        opacity=1.0,
                        line=dict(width=0),
                    ),
                    name=info["name"],
                    customdata=custom,
                    hovertemplate=(
                        "<b>Pred Raw:</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                        "<b>Pred Canon:</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]})<br>"
                        "<b>Label:</b> (%{customdata[6]}, %{customdata[7]}, %{customdata[8]})<br>"
                        "<b>Intensity:</b> %{customdata[9]:.2f}<br>"
                        "<b>Coord:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
                    ),
                    visible=visible,
                )
            )
        elif info["type"] == "flow":
            if key == "gt_flow":
                hkl_data = vis_data["labels"]
            elif key == "pred_flow_canon" and "predictions_canon" in vis_data:
                hkl_data = vis_data["predictions_canon"]
            else:
                hkl_data = vis_data.get("predictions_raw", vis_data["labels"])
            points = vis_data["p0_original"]
            intensities = vis_data["intensities"]
            hkl_vectors = hkl_data.astype(np.float32)
            norms = np.linalg.norm(hkl_vectors, axis=1)
            non_zero_mask = norms > 1e-6
            zero_points = points[~non_zero_mask]
            traces.append(
                go.Scatter3d(
                    x=zero_points[:, 0],
                    y=zero_points[:, 1],
                    z=zero_points[:, 2],
                    mode="markers",
                    marker=dict(color="grey", size=2, opacity=0.6),
                    hoverinfo="skip",
                    visible=visible,
                    name=f"{info['name']} (zero hkl)",
                )
            )
            if np.any(non_zero_mask):
                start_points = points[non_zero_mask]
                directions = hkl_vectors[non_zero_mask] / norms[non_zero_mask, np.newaxis]
                intensities_nz = intensities[non_zero_mask]
                lengths = 0.00 + 0.1 * (
                    (intensities_nz - intensities_nz.min())
                    / (intensities_nz.max() - intensities_nz.min() + 1e-6)
                )
                end_points = start_points + directions * lengths[:, np.newaxis]
                lines_x, lines_y, lines_z = [], [], []
                for i in range(len(start_points)):
                    lines_x.extend([start_points[i, 0], end_points[i, 0], None])
                    lines_y.extend([start_points[i, 1], end_points[i, 1], None])
                    lines_z.extend([start_points[i, 2], end_points[i, 2], None])
                intensities_repeat = np.repeat(intensities_nz, 3)
                traces.append(
                    go.Scatter3d(
                        x=lines_x,
                        y=lines_y,
                        z=lines_z,
                        mode="lines",
                        line=dict(
                            width=0.4,
                            color=intensities_repeat,
                            colorscale="Bluered",
                            cmin=intensities_nz.min(),
                            cmax=intensities_nz.max(),
                        ),
                        customdata=intensities_repeat,
                        hovertemplate="<b>Intensity:</b> %{customdata:.2f}<extra></extra>",
                        visible=visible,
                        name=f"{info['name']} (vectors)",
                    )
                )
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible))
        elif info["type"] == "structure":
            if key in vis_data:
                points = vis_data[key]
                intns = vis_data["intensities"]
                traces.append(
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode="markers",
                        marker=dict(size=intensity_to_size(intns), color=info["color"], opacity=1.0, line=dict(width=0.1),),
                        name=f"{info['name']} ({len(points)} points)",
                        visible=visible,
                    )
                )
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], visible=visible))

    for tr in traces:
        fig.add_trace(tr)

    buttons = []
    trace_counter = 0
    for key, info in stages.items():
        visibility = [False] * len(traces)
        num_traces = 2 if info["type"] == "flow" else 1
        for i in range(num_traces):
            if trace_counter + i < len(traces):
                visibility[trace_counter + i] = True
        trace_counter += num_traces
        buttons.append(
            dict(
                label=info["name"],
                method="update",
                args=[{"visible": visibility}, {"title": f"Current: {info['name']}"}],
            )
        )

    fig.update_layout(
        title_text="Current: Ground Truth (color)",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        scene=hidden_axes_scene("data"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=1),
    )

if __name__ == "__main__":
    main()


