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


# MODEL_PATH = "/media/max/Data/results/xrd_transformer/mp_random_150k_canonical/XRDT_20250918_010050/best_model.pth"
MODEL_PATH = "pretrained/best_model_v123_angle_limited.pth"
DATA_FILE = "/media/max/Data/datasets/mp_random_150k_v3_canonical/test/test_000020.jsonl"
NUM_CLASSES = 11
IN_CHANNELS = 4
VIS_MASKING_RATIO = 1.0


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


def run_and_capture_with_hooks(
    model: torch.nn.Module,
    coords_xyz: torch.Tensor,
    features: torch.Tensor,
    offsets: torch.Tensor,
    original_labels: torch.Tensor,
    original_intensities: torch.Tensor,
    masked_indices: torch.Tensor,
    is_abs_label: bool,
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
        vis_data["predictions"] = predictions.detach().cpu().numpy() - miller_offset_val
        vis_data["labels"] = original_labels.detach().cpu().numpy() - miller_offset_val
        vis_data["intensities"] = original_intensities.detach().cpu().numpy()
        vis_data["masked_indices"] = masked_indices.detach().cpu().numpy()

    return vis_data


def add_main_figure(fig: go.Figure, vis_data: dict, is_abs_label: bool):
    stages = OrderedDict(
        [
            ("gt_color", {"name": "Ground Truth (color)", "type": "color"}),
            ("pred_color", {"name": "Prediction (color)", "type": "color"}),
            ("gt_flow", {"name": "Ground Truth (direction)", "type": "flow"}),
            ("pred_flow", {"name": "Prediction (direction)", "type": "flow"}),
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
            hkl_data = vis_data["labels"] if "gt" in key else vis_data["predictions"]
            points = vis_data["p0_original"]
            intns = vis_data["intensities"]
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
                    customdata=np.hstack((hkl_data, intns[:, np.newaxis])),
                    hovertemplate=(
                        "<b>hkl:</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                        "<b>Intensity:</b> %{customdata[3]:.2f}<br>"
                        "<b>Coord:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
                    ),
                    visible=visible,
                )
            )
        elif info["type"] == "flow":
            hkl_data = vis_data["labels"] if "gt" in key else vis_data["predictions"]
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


def main():
    parser = argparse.ArgumentParser(description="Visualize PointTransformerV3 intermediate outputs and predictions.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", default=DATA_FILE, help="Path to a jsonl sample file")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes")
    parser.add_argument("--in-channels", type=int, default=IN_CHANNELS, help="Number of input channels")
    parser.add_argument("--masking-ratio", type=float, default=VIS_MASKING_RATIO, help="Masking ratio for visualization")
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda, cuda:0, cpu). Default: auto")
    parser.add_argument("--outdir", default="vis_mask_outputs", help="Directory to save HTML visualizations")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    # import pointcept

    from XRDT.model import XRDT

    model_path = args.model
    data_file = args.data
    num_classes = args.num_classes
    in_channels = args.in_channels
    vis_mask_ratio = float(args.masking_ratio)

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
        hkl_features_unmasked = torch.abs(labels_raw).clone().float() / 5.0
    else:
        hkl_features_unmasked = labels_raw.clone().float() / 5.0

    num_points = coords_feats.shape[0]
    num_to_mask = int(num_points * vis_mask_ratio)
    perm = torch.randperm(num_points)
    masked_indices = perm[:num_to_mask]
    hkl_features_masked = hkl_features_unmasked.clone()
    if num_to_mask > 0:
        hkl_features_masked[masked_indices, :] = 0.0

    # Intensity normalization and minor coordinate normalization per notebook
    coords_feats[:, 3] /= 10.0
    coords_feats[:, 1:3] = (
        (coords_feats[:, 1:3] - 0.5) * 0.99 / torch.max(coords_feats[:, 1:3]) + 0.5
    )

    if in_channels == 4:
        feats_with_hkl = coords_feats
    else:
        feats_with_hkl = torch.cat([coords_feats, hkl_features_masked], dim=1)

    if is_abs:
        labels_final = torch.abs(labels_raw)
    else:
        labels_final = labels_raw + miller_offset

    offsets = torch.tensor([len(coords_only)], dtype=torch.long)
    original_labels_tensor = labels_final
    original_intensities = coords_feats[:, 3] * 10.0

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
        masked_indices,  # keep on CPU
        is_abs_label=is_abs,
    )
    print("--> Capture done.")

    # Report metrics on masked points only
    print("\n--- Metrics on masked points only ---")
    masked_preds = vis_data["predictions"][vis_data["masked_indices"]]
    masked_labels = vis_data["labels"][vis_data["masked_indices"]]
    num_masked_points = len(masked_labels)
    if num_masked_points > 0:
        h_correct = (masked_preds[:, 0] == masked_labels[:, 0]).sum()
        k_correct = (masked_preds[:, 1] == masked_labels[:, 1]).sum()
        l_correct = (masked_preds[:, 2] == masked_labels[:, 2]).sum()
        all_correct = ((masked_preds == masked_labels).all(axis=1)).sum()
        print(f"Masked points: {num_masked_points}")
        print(f"H accuracy: {h_correct / num_masked_points * 100:.2f}%")
        print(f"K accuracy: {k_correct / num_masked_points * 100:.2f}%")
        print(f"L accuracy: {l_correct / num_masked_points * 100:.2f}%")
        print(f"HKL exact match: {all_correct / num_masked_points * 100:.2f}%")
    else:
        print("No masked points; metrics skipped.")
    print("-" * 40)

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

    # Figure 2: Mask distribution
    all_points = vis_data["p0_original"]
    masked_idx_np = vis_data["masked_indices"]
    unmasked_indices = np.setdiff1d(np.arange(len(all_points)), masked_idx_np)
    masked_points = all_points[masked_idx_np]
    unmasked_points = all_points[unmasked_indices]

    print(len(unmasked_points), len(masked_points))

    fig_mask = go.Figure()
    fig_mask.add_trace(
        go.Scatter3d(
            x=unmasked_points[:, 0],
            y=unmasked_points[:, 1],
            z=unmasked_points[:, 2],
            mode="markers",
            marker=dict(size=1, color="black", opacity=1),
            name=f"Unmasked Points ({len(unmasked_points)})",
        )
    )
    fig_mask.add_trace(
        go.Scatter3d(
            x=masked_points[:, 0],
            y=masked_points[:, 1],
            z=masked_points[:, 2],
            mode="markers",
            marker=dict(size=0.3, color="white"),
            name=f"Masked Points ({len(masked_points)})",
        )
    )
    fig_mask.update_layout(
        title="Masked points distribution",
        scene=hidden_axes_scene("data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    mask_path = os.path.join(outdir, "mask_distribution.html")
    fig_mask.write_html(mask_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {os.path.abspath(mask_path)}")
    if open_in_browser:
        try:
            webbrowser.open(f"file://{os.path.abspath(mask_path)}")
        except Exception as e:
            print(f"Failed to open browser: {e}")

    # Figure 3: HKL difference visualization
    preds = vis_data["predictions"]
    labels = vis_data["labels"]
    points = vis_data["p0_original"]

    # Keep the same metric as the notebook (sum of abs, negated)
    l2_diff = np.sum(np.abs(preds - labels), axis=1) * -1

    fig_diff = go.Figure()
    fig_diff.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=1,
                color=l2_diff,
                colorscale="Reds_r",
                cmin=0,
                cmax=l2_diff.max(),
                colorbar=dict(title="HKL L1 Difference"),
                line=dict(width=0),
            ),
            customdata=np.hstack((preds, labels, l2_diff[:, np.newaxis])),
            hovertemplate=(
                "<b>Pred:</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>"
                "<b>Label:</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]})<br>"
                "<b>Diff:</b> %{customdata[6]:.2f}<br>"
                "<b>Coord:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"
            ),
            name="HKL Difference",
        )
    )
    fig_diff.update_layout(
        title="HKL difference (L1 distance)",
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


if __name__ == "__main__":
    main()


