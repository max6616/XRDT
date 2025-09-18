import torch
import torch.nn as nn
from functools import partial
from addict import Dict
import math
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

import flash_attn
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)
        return out

class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        patch_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and flash_attn is not None
        if self.enable_flash:
            assert not enable_rpe
            assert not upcast_attention
            assert not upcast_softmax
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key, unpad_key, cu_seqlens_key = "pad", "unpad", "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc")
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i+1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i+1]] = pad[_offset_pad[i+1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i+1] - self.patch_size]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(_offset_pad[i], _offset_pad[i + 1], step=self.patch_size, dtype=torch.int32, device=offset.device)
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        
        H, K, C = self.num_heads, self.patch_size, self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H), cu_seqlens, self.patch_size,
                dropout_p=self.attn_drop if self.training else 0, softmax_scale=self.scale
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Block(PointModule):
    def __init__(
        self, channels, num_heads, patch_size=48, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
        pre_norm=True, order_index=0, cpe_indice_key=None, enable_rpe=False, enable_flash=True,
        upcast_attention=True, upcast_softmax=True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels, patch_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, order_index,
            enable_rpe, enable_flash, upcast_attention, upcast_softmax
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(channels, int(channels * mlp_ratio), channels, act_layer, proj_drop)
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm: point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm1(point)
        shortcut = point.feat
        if self.pre_norm: point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm: point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class SerializedPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None, act_layer=None, reduce="max", shuffle_orders=False, traceable=True):
        super().__init__()
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        assert reduce in ["sum", "mean", "min", "max"]
        self.stride, self.reduce, self.shuffle_orders, self.traceable = stride, reduce, shuffle_orders, traceable
        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None: self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None: self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth: pooling_depth = 0
        assert {"serialized_code", "serialized_order", "serialized_inverse", "serialized_depth"}.issubset(point.keys())
        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(1, order, torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, order, inverse = code[perm], order[perm], inverse[perm]
        
        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            # coord=point.coord[head_indices],
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code, serialized_order=order, serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth, batch=point.batch[head_indices],
        )
        if self.traceable:
            point_dict["pooling_inverse"], point_dict["pooling_parent"] = cluster, point
        point = Point(point_dict)
        if hasattr(self, 'norm'): point = self.norm(point)
        if hasattr(self, 'act'): point = self.act(point)
        point.sparsify()
        return point

class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj, self.proj_skip = PointSequential(nn.Linear(in_channels, out_channels)), PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels)), self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer()), self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        if self.traceable: parent["unpooling_parent"] = point
        return parent

class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem")
        )
        if norm_layer is not None: self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None: self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)

class PointTransformerV3(PointModule):
    def __init__(
        self, in_channels=6, order=("z", "z-trans"), stride=(2, 2, 2, 2),
        enc_depths=(4, 6, 4, 3, 2), enc_channels=(64, 128, 256, 512, 1024),
        enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(4, 4, 4, 4, 4), dec_channels=(64, 128, 256, 512),
        dec_num_head=(2, 4, 8, 16), dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.0,
        pre_norm=True, shuffle_orders=True, enable_rpe=False, enable_flash=True,
        upcast_attention=False, upcast_softmax=False, cls_mode=False
    ):
    # def __init__(
    #     self, in_channels=6, order=("z", "z-trans"), stride=(2, 2, 2, 2),
    #     enc_depths=(2, 2, 2, 2, 2), enc_channels=(16, 32, 64, 128, 256),
    #     enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48, 48, 48, 48, 48),
    #     dec_depths=(2, 2, 2, 2), dec_channels=(32, 32, 64, 128),
    #     dec_num_head=(2, 4, 8, 16), dec_patch_size=(48, 48, 48, 48),
    #     mlp_ratio=2, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
    #     pre_norm=True, shuffle_orders=True, enable_rpe=False, enable_flash=True,
    #     upcast_attention=False, upcast_softmax=False, cls_mode=False
    # ):
        super().__init__()
        self.num_stages, self.order, self.cls_mode, self.shuffle_orders = len(enc_depths), [order] if isinstance(order, str) else order, cls_mode, shuffle_orders
        
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels, enc_channels[0], ln_layer, act_layer)
        
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]):sum(enc_depths[:s+1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(enc_channels[s-1], enc_channels[s], stride[s-1], ln_layer, act_layer), name="down")
            for i in range(enc_depths[s]):
                enc.add(Block(
                    enc_channels[s], enc_num_head[s], enc_patch_size[s], mlp_ratio, qkv_bias, qk_scale,
                    attn_drop, proj_drop, enc_drop_path_[i], ln_layer, act_layer, pre_norm, i % len(self.order),
                    f"stage{s}", enable_rpe, enable_flash, upcast_attention, upcast_softmax
                ), name=f"block{i}")
            if len(enc) != 0: self.enc.add(enc, name=f"enc{s}")

        if not self.cls_mode:
            self.dec = PointSequential()
            dec_channels_list = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages-1)):
                dec = PointSequential()
                dec.add(SerializedUnpooling(dec_channels_list[s+1], enc_channels[s], dec_channels[s], ln_layer, act_layer), name="up")
                for i in range(dec_depths[s]):
                    dec.add(Block(
                        dec_channels[s], dec_num_head[s], dec_patch_size[s], mlp_ratio, qkv_bias, qk_scale,
                        attn_drop, proj_drop, 0, ln_layer, act_layer, pre_norm, i % len(self.order),
                        f"stage{s}", enable_rpe, enable_flash, upcast_attention, upcast_softmax
                    ), name=f"block{i}")
                self.dec.add(dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        encoder_output = self.enc(point)
        if not self.cls_mode: 
            true_encoder_features = encoder_output.feat.clone()
            true_encoder_batch = encoder_output.batch.clone()
            true_encoder_coord = encoder_output.coord.clone()
            
            decoder_output = self.dec(encoder_output)
            
            encoder_dict = {
                'feat': true_encoder_features,
                'batch': true_encoder_batch, 
                'coord': true_encoder_coord
            }
            true_encoder_output = Point(encoder_dict)
            
            return {"encoder_output": true_encoder_output, "decoder_output": decoder_output}
        return encoder_output

class XRDT(nn.Module):
    def __init__(self, in_channels=7, num_classes=11, **kwargs):
        super().__init__()
        
        decoder_out_channels = kwargs.get("dec_channels", (64, 128, 256, 512))[0]
        
        encoder_out_channels = kwargs.get("enc_channels", (64, 128, 256, 512, 1024))[-1]
        
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            cls_mode=False,
            **kwargs
        )

        self.cls_h = self._make_head(decoder_out_channels, num_classes)
        self.cls_k = self._make_head(decoder_out_channels, num_classes)
        self.cls_l = self._make_head(decoder_out_channels, num_classes)
        
        pooled_feature_dim = encoder_out_channels
        
        self.reg_lattice = nn.Sequential(
            nn.Linear(pooled_feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 6)
        )
        
        self.cls_space_group = nn.Sequential(
            nn.Linear(pooled_feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 230)
        )

    def _make_head(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Linear(in_planes, in_planes // 2), 
            nn.LayerNorm(in_planes // 2), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_planes // 2, out_planes)
        )

    def forward(self, p, x, o):
        data_dict = {"coord": p, "feat": x, "offset": o, "grid_size": 0.01}
        
        backbone_out = self.backbone(data_dict)
        encoder_features = backbone_out["encoder_output"].feat # (N, C_enc)
        decoder_features = backbone_out["decoder_output"].feat # (N, C_dec)
        
        h = self.cls_h(decoder_features)
        k = self.cls_k(decoder_features)
        l = self.cls_l(decoder_features)
        
        encoder_batch = backbone_out["encoder_output"].batch
        
        pooled_encoder_features = torch_scatter.scatter_mean(encoder_features, encoder_batch, dim=0)
        
        lattice_params = self.reg_lattice(pooled_encoder_features)
        space_group = self.cls_space_group(pooled_encoder_features)
        
        return {
            'h': h, 
            'k': k, 
            'l': l,
            'lattice_params': lattice_params,
            'space_group': space_group
        }
