#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import random
from pathlib import Path
import math
import json
import csv
import inspect

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import models.convnextv2 as convnextv2
import models.convnextv1 as convnextv1

try:
    import models.timm_backbones as timm_backbones
except Exception:
    timm_backbones = None

try:
    import models.tv_backbones as tv_backbones
except Exception:
    tv_backbones = None

from engine_finetune import train_one_epoch, evaluate, SimpleEMA
import utils as U
import datasets as D
from models.convnextv2_dual import ConvNeXtV2Dual
from torch.optim.lr_scheduler import LambdaLR
class GSPFRegularizedCriterion(nn.Module):
    def __init__(
        self,
        base_criterion,
        lambda_consistency=0.0,
        lambda_ortho=0.0,
        cv_lambda_sem=0.0,
        cv_lambda_geo=0.0,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_ortho = float(lambda_ortho)
        self.cv_lambda_sem = float(cv_lambda_sem)
        self.cv_lambda_geo = float(cv_lambda_geo)

    def forward(self, logits, targets):
        base_loss = self.base_criterion(logits, targets)

        from models.modules.attentions import pop_gspf_regularization, reset_gspf_cache

        try:
            from models.modules.cross_view_consistency import pop_cv_consistency, reset_cv_cache
            has_cv = True
        except Exception:
            has_cv = False

        total_loss = base_loss

        # GSPF regularization
        if self.lambda_consistency <= 0 and self.lambda_ortho <= 0:
            reset_gspf_cache()
        else:
            reg_loss = pop_gspf_regularization(
                lambda_consistency=self.lambda_consistency,
                lambda_ortho=self.lambda_ortho,
            )
            if not isinstance(reg_loss, float):
                total_loss = total_loss + reg_loss

        # Cross-view consistency regularization
        if has_cv:
            if self.cv_lambda_sem <= 0 and self.cv_lambda_geo <= 0:
                reset_cv_cache()
            else:
                cv_loss = pop_cv_consistency(
                    lambda_sem=self.cv_lambda_sem,
                    lambda_geo=self.cv_lambda_geo,
                )
                if not isinstance(cv_loss, float):
                    total_loss = total_loss + cv_loss

        return total_loss

def build_base_criterion(args) -> nn.Module:
    """根据 --base_loss 选择基础监督损失；保持多标签场景默认 BCE。"""
    if args.base_loss == 'bce':
        return nn.BCEWithLogitsLoss()
    elif args.base_loss == 'mlsm':
        return nn.MultiLabelSoftMarginLoss()

    elif args.base_loss == 'focal':
        from models.modules.custom_losses.focal_loss import FocalLoss
        return FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)

    elif args.base_loss == 'asl':
        from models.modules.custom_losses.asymmetric_loss import AsymmetricLossMultiLabel
        return AsymmetricLossMultiLabel(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
        )

    elif args.base_loss == 'fals':
        from models.modules.custom_losses.fals_loss import FALSLoss
        return FALSLoss(eps=args.fals_eps, gamma=args.fals_gamma, reduction='mean')

    elif args.base_loss == 'mcb':
        from models.modules.custom_losses.mcb_loss import MCBLoss
        return MCBLoss(momentum=args.mcb_momentum, reduction='mean')

    elif args.base_loss == 'dals':
        from models.modules.custom_losses.dals_loss import DALSBCE
        return DALSBCE(eps=args.dals_eps, gamma=args.dals_gamma)

    elif args.base_loss == 'mcb_convex':
        from models.modules.custom_losses.mcb_loss import MCBLossConvex
        return MCBLossConvex(tau=args.mcb_tau, w_min=args.mcb_wmin, momentum=args.mcb_momentum)

    elif args.base_loss == 'gebce':
        from models.modules.custom_losses.gebce import GEBCELoss
        return GEBCELoss(
            lambda_coef=args.ge_lambda,
            pos_only=args.ge_pos_only,
            alpha=args.ge_alpha,
            ema=args.ge_ema,
            momentum=args.ge_momentum,
            band=args.ge_band,
            trainable=args.ge_trainable,
        )

    else:
        return nn.BCEWithLogitsLoss()


def resolve_attention_config(name):
    ATTN_MAP = {
        # ===== 原有 =====
        "granularity": {"N3": "granularity", "N4": "granularity", "N5": "granularity"},
        "proto_route": {"N3": "proto_route", "N4": "proto_route", "N5": "proto_route"},
        "freq_route": {"N3": "freq_route", "N4": "freq_route", "N5": "freq_route"},
        "polarity": {"N3": "polarity", "N4": "polarity", "N5": "polarity"},
        "self_feedback": {"N3": "self_feedback", "N4": "self_feedback", "N5": "self_feedback"},

        # ===== 你之前已有 =====
        "n3_freq_n4_proto_n5_gran": {
            "N3": "freq_route", "N4": "proto_route", "N5": "granularity"
        },
        "parallel_freqgran_n4_proto_n5_selffb": {
            "N3": {"parallel": ["freq_route", "granularity"]},
            "N4": "proto_route",
            "N5": "self_feedback",
        },
        "seq_freqpol_n4_proto_n5_selffb": {
            "N3": ["freq_route", "polarity"],
            "N4": "proto_route",
            "N5": "self_feedback",
        },

        # ===== ✅ 你新加的12组 =====
        "n3_gran_n4_gran_n5_proto": {
            "N3": "granularity", "N4": "granularity", "N5": "proto_route"
        },
        "n3_gran_n4_proto_n5_proto": {
            "N3": "granularity", "N4": "proto_route", "N5": "proto_route"
        },
        "n3_gran_n4_proto_n5_gran": {
            "N3": "granularity", "N4": "proto_route", "N5": "granularity"
        },
        "n3_proto_n4_gran_n5_proto": {
            "N3": "proto_route", "N4": "granularity", "N5": "proto_route"
        },
        "n3_proto_n4_proto_n5_gran": {
            "N3": "proto_route", "N4": "proto_route", "N5": "granularity"
        },
        "parallel_gran_proto_n4_proto_n5_gran": {
            "N3": {"parallel": ["granularity", "proto_route"]},
            "N4": "proto_route",
            "N5": "granularity",
        },
        "n3_gran_n4_parallel_gran_proto_n5_proto": {
            "N3": "granularity",
            "N4": {"parallel": ["granularity", "proto_route"]},
            "N5": "proto_route",
        },
        "parallel_freq_gran_n4_gran_n5_proto": {
            "N3": {"parallel": ["freq_route", "granularity"]},
            "N4": "granularity",
            "N5": "proto_route",
        },
        "parallel_gran_proto_n4_gran_n5_proto": {
            "N3": {"parallel": ["granularity", "proto_route"]},
            "N4": "granularity",
            "N5": "proto_route",
        },
        "n3_gran_n4_parallel_freq_gran_n5_proto": {
            "N3": "granularity",
            "N4": {"parallel": ["freq_route", "granularity"]},
            "N5": "proto_route",
        },
        "n3_gran_n4_proto_n5_none": {
            "N3": "granularity", "N4": "proto_route", "N5": None
        },
        "n3_none_n4_proto_n5_gran": {
            "N3": None, "N4": "proto_route", "N5": "granularity"
        },

        # ==============================
        # GSPF version study
        # ==============================

        # E0: 当前强基线
        "baseline_gran_proto_gran": {
            "N3": "granularity",
            "N4": "proto_route",
            "N5": "granularity",
        },

        # V1: 只把 N4 的固定 Proto 换成动态 Proto
        "v1_gran_dynproto_gran": {
            "N3": "granularity",
            "N4": "dynamic_proto",
            "N5": "granularity",
        },

        # V2: 只在 N4 使用 GSPF，最推荐先跑这个
        "v2_gran_gspf_gran": {
            "N3": "granularity",
            "N4": "gspf",
            "N5": "granularity",
        },

        # V2-full-arch: 全层都用 GSPF
        "v2_gspf_all": {
            "N3": "gspf",
            "N4": "gspf",
            "N5": "gspf",
        },

        # V2-alt: N3/N5 用 GSPF，N4 保留 Proto
        "v2_gspf_proto_gspf": {
            "N3": "gspf",
            "N4": "proto_route",
            "N5": "gspf",
        },

        "v3_gran_gspf_gran_cons": {
            "N3": "granularity",
            "N4": "gspf_reg",
            "N5": "granularity",
        },

        # Full: 结构和 V2 一样，区别是命令里同时打开 consistency + ortho loss
        "full_gran_gspf_gran": {
            "N3": "granularity",
            "N4": "gspf_reg",
            "N5": "granularity",
        },

        # Proto 主导 + GSPF 小残差增强
        "v2_gran_proto_gspfres_gran": {
            "N3": "granularity",
            "N4": "proto_gspf_residual",
            "N5": "granularity",
        },

        "v2_gran_proto_gspfres_none": {
            "N3": "granularity",
            "N4": "proto_gspf_residual",
            "N5": None,
        },

        "v2_none_proto_gspfres_gran": {
            "N3": None,
            "N4": "proto_gspf_residual",
            "N5": "granularity",
        },

        # ==============================
        # Weak-Class Guided Attention
        # ==============================

        # W1：只在 N4 做弱类原型调制，N3 保持普通 Gran，N5 不加注意力
        "w1_gran_wcproto_none": {
            "N3": "granularity",
            "N4": "wc_proto_route",
            "N5": None,
        },

        # W2：N3 做弱类粒度调制，N4 做弱类原型调制，N5 不加注意力
        # 这是后续最值得主推的结构
        "w2_wcgran_wcproto_none": {
            "N3": "wc_granularity",
            "N4": "wc_proto_route",
            "N5": None,
        },

        # W3：只改 N3，N4 仍用普通 Proto，用于消融
        "w3_wcgran_proto_none": {
            "N3": "wc_granularity",
            "N4": "proto_route",
            "N5": None,
        },

        # W4：弱类调制但保留 N5 Gran，看 N5 是否继续干扰
        "w4_wcgran_wcproto_gran": {
            "N3": "wc_granularity",
            "N4": "wc_proto_route",
            "N5": "granularity",
        },

        # ==============================
        # DWR: Difficulty-aware Weak-class Expert Routing
        # ==============================

        # DWR 主推结构：
        # N3 保留稳定 Granularity，N4 使用动态弱类专家路由，N5 不加注意力
        "dwr_gran_dwr_none": {
            "N3": "granularity",
            "N4": "dwr_route",
            "N5": None,
        },

        # 消融1：DWR + N5 Gran，看高层注意力是否有帮助
        "dwr_gran_dwr_gran": {
            "N3": "granularity",
            "N4": "dwr_route",
            "N5": "granularity",
        },

        # 消融2：N3 也使用弱类粒度调制
        "dwr_wcgran_dwr_none": {
            "N3": "wc_granularity",
            "N4": "dwr_route",
            "N5": None,
        },


        # ==============================
        # Liquid Adapter / LG-v2: R2-preserving liquid variants
        # ==============================
        "ladapter_gran_ladapter_none": {
            "N3": "granularity",
            "N4": "liquid_adapter_pgspr",
            "N5": None,
        },

        # Weak-LAdapter: R2 + Liquid Adapter + weak-class guided channel scaling
        "weak_ladapter_gran_wladapter_none": {
            "N3": "granularity",
            "N4": "weak_liquid_adapter_pgspr",
            "N5": None,
        },

        "wladapter_gran_wladapter_none": {
            "N3": "granularity",
            "N4": "weak_liquid_adapter_pgspr",
            "N5": None,
        },

        "lgv2_gran_lg_none": {
            "N3": "granularity",
            "N4": "lg_pgspr_v2",
            "N5": None,
        },

        "lgv2_gran_lg_gran": {
            "N3": "granularity",
            "N4": "lg_pgspr_v2",
            "N5": "granularity",
        },

        # ==============================
        # DWR-V2: R2-dominant dual-expert routing
        # ==============================

        # 主推版本：N3 保留 Gran，N4 使用 DWR-V2，N5 不加注意力
        "dwr2_gran_dwr2_none": {
            "N3": "granularity",
            "N4": "dwr2_route",
            "N5": None,
        },

        # 消融版本：DWR-V2 + N5 Gran
        "dwr2_gran_dwr2_gran": {
            "N3": "granularity",
            "N4": "dwr2_route",
            "N5": "granularity",
        },




    }

    if name not in ATTN_MAP:
        raise ValueError(
            f"未知的 --attention_name: {name}. 可选值: {', '.join(sorted(ATTN_MAP.keys()))}"
        )

    return ATTN_MAP[name]


def append_summary_to_global_log(args, best_metric_value, metric_name, model_total_params,
                                 class_names, per_class_ap_list):
    """
    将本次实验的最终总结（包含 per-class AP），追加写入到全局日志文件中。
    """
    summary_file_path = Path(getattr(args, "summary_csv", "viewaware_llm_5seed_detailed.csv"))
    summary_file_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        'output_dir', 'model', 'aug_mode', 'fuse_mode', 'ahcr_mode',
        'attention_name', 'attention_config',
        'best_metric_name', 'best_metric_value', 'total_params_M',
        'batch_size', 'learning_rate',
    ]

    if class_names and per_class_ap_list:
        ap_headers = [f"AP_{name.replace(' ', '_')}" for name in class_names]
        headers.extend(ap_headers)

    summary_data = {
        'output_dir': args.output_dir,
        'model': args.model,
        'aug_mode': args.aug_mode,
        'fuse_mode': args.fuse_mode,
        'ahcr_mode': args.ahcr_mode if args.fuse_mode == 'ahcr' else 'N/A',
        'attention_name': getattr(args, 'attention_name', '') if getattr(args, 'attention_name', None) else '',
        'attention_config': (
            json.dumps(resolve_attention_config(args.attention_name), ensure_ascii=False)
            if getattr(args, 'attention_name', None)
            else (args.attention_config if args.attention_config else '{}')
        ),
        'best_metric_name': metric_name,
        'best_metric_value': f"{best_metric_value:.4f}",
        'total_params_M': f"{model_total_params / 1_000_000:.2f}",
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }

    if class_names and per_class_ap_list and len(class_names) == len(per_class_ap_list):
        for i, name in enumerate(class_names):
            summary_data[f"AP_{name.replace(' ', '_')}"] = f"{per_class_ap_list[i]:.4f}"

    try:
        file_exists = summary_file_path.is_file()
        with open(summary_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_data)
        print(f"📈 最终结果（含Per-Class AP）已成功追加到总成绩表: {summary_file_path}")
    except Exception as e:
        print(f"❌ 写入总成绩表时发生错误: {e}")


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_finetune_weights(model: nn.Module, ckpt_path: str, prefix: str = ''):
    if not ckpt_path:
        return None
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[finetune] file not found: {ckpt_path}")
        return None

    sd = None
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        obj = torch.load(str(p), map_location="cpu")
        sd = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if prefix and nk.startswith(prefix):
            nk = nk[len(prefix):]
        cleaned[nk] = v

    try:
        mount = getattr(args, "model_mount", "")
    except NameError:
        mount = ""
    if mount:
        cleaned = {(mount + k): v for k, v in cleaned.items()}

    msd = model.state_dict()
    to_load = {}
    skipped_shape = []
    for k, v in cleaned.items():
        if k in msd and tuple(v.shape) == tuple(msd[k].shape):
            to_load[k] = v
        elif k in msd:
            skipped_shape.append((k, tuple(v.shape), tuple(msd[k].shape)))

    msg = model.load_state_dict(to_load, strict=False)
    print(f"[finetune] loaded={len(to_load)}  skipped_shape={len(skipped_shape)}  "
          f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")

    _missing = list(getattr(msg, 'missing_keys', []))
    _unexpected = list(getattr(msg, 'unexpected_keys', []))

    if _missing:
        print("[finetune] missing keys (first 200):")
        for k in _missing[:200]:
            print("  -", k)

    if _unexpected:
        print("[finetune] unexpected keys (first 200):")
        for k in _unexpected[:200]:
            print("  -", k)

    out_dir = os.environ.get("OUTPUT_DIR_HINT", "")
    try:
        out_dir = out_dir or getattr(globals().get('args', None), 'output_dir', '')
    except Exception:
        pass

    save_root = out_dir if out_dir else "."
    try:
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "finetune_missing_keys.txt"), "w") as f:
            for k in _missing:
                f.write(k + "\n")
        with open(os.path.join(save_root, "finetune_unexpected_keys.txt"), "w") as f:
            for k in _unexpected:
                f.write(k + "\n")
        print(f"[finetune] 已将缺失/意外键清单写入到: {save_root}/finetune_*_keys.txt")
    except Exception as e:
        print(f"[finetune] ⚠️ 保存缺失/意外键清单失败: {e}")

    if skipped_shape:
        print("[finetune] first few shape-mismatch keys:")
        for i, (k, s_ckpt, s_model) in enumerate(skipped_shape[:10]):
            print(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")
    return cleaned


def _try_build_loaders_with_project(args):
    builder_names = ["build_loaders", "build_dataloaders", "create_loaders", "create_dataloaders"]
    for name in builder_names:
        if hasattr(D, name):
            return getattr(D, name)(args)
    if hasattr(D, "XrayMultiLabelList"):
        train_ds = D.XrayMultiLabelList(
            args.train_list, args.classes_file, is_train=True,
            dual_view=args.dual_view, input_size=args.input_size
        )
        val_ds = D.XrayMultiLabelList(
            args.val_list, args.classes_file, is_train=False,
            dual_view=args.dual_view, input_size=args.input_size
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context='spawn',
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
        return train_loader, val_loader, getattr(train_ds, "class_names", None)
    raise RuntimeError("datasets.py 缺少构建函数（build_loaders/...），请保留工程里的数据集逻辑。")


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--model', default='convnextv2_base')
    parser.add_argument(
        '--official_ahcr_source_dir',
        default='third_party/DvXray_official',
        help='Read-only checkout of the locked official DvXray/AHCR source.',
    )
    parser.add_argument(
        '--official_ahcr_source_commit',
        default='a6bfc1b1299d28e8226c106a94967287a8e30927',
        help='Exact official DvXray/AHCR commit required by the adapter.',
    )
    parser.add_argument(
        '--official_ahcr_pretrained_weights',
        default='IMAGENET1K_V2',
        choices=['IMAGENET1K_V1', 'IMAGENET1K_V2', 'NONE'],
        help='ResNet50 initialization; V2 matches the local Plain-BCE baseline.',
    )
    parser.add_argument('--input_size', default=384, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation steps; 1 preserves legacy behavior')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--drop_path', default=0.2, type=float)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--finetune', default='')
    parser.add_argument('--model_prefix', default='', help="加前缀")
    parser.add_argument("--model_mount", type=str, default="", help="可选：统一挂在到某子模块前")

    parser.add_argument('--dual_view', default=True, type=U.str2bool)
    parser.add_argument(
        '--view_mode',
        default='paired',
        choices=['paired', 'a_only', 'b_only'],
        help='Input-view ablation: paired=A+B, a_only=A+A, b_only=B+B.',
    )
    parser.add_argument('--train_list', default='annotations/DvXray_train.txt')
    parser.add_argument('--val_list', default='annotations/DvXray_val.txt')
    parser.add_argument('--classes_file', default='annotations/classes.txt')
    parser.add_argument('--num_classes', default=15, type=int)
    parser.add_argument('--multi_label', default=True, type=U.str2bool)
    parser.add_argument('--eval_threshold', default=0.5, type=float)

    parser.add_argument('--return_intermediate', default=False, type=U.str2bool)
    parser.add_argument('--out_indices', default=[1, 2, 3], nargs='+', type=int)

    parser.add_argument('--fuse_mode', default='concat', choices=['concat', 'add', 'mean', 'max', 'gated', 'xattn', 'ahcr', 'cv_gsc_add'])
    parser.add_argument('--fuse_levels', default=['C3', 'C4', 'C5'], nargs='+')
    parser.add_argument('--head_type', default='c5', type=str,
                        help="Type of head for feature aggregation: c5, fpn, fpn_fuse, fpn_pan")

    parser.add_argument('--attention_name', type=str, default=None,
                        help='Safe attention config name, e.g. polarity / granularity / proto_route')
    parser.add_argument('--attention_config', type=str, default=None,
                        help='[Legacy] JSON string for complex attention configurations. Prefer --attention_name.')

    parser.add_argument('--xattn_heads', default=4, type=int, help='仅 fuse_mode=xattn 时使用')
    parser.add_argument('--xattn_reduction', default=4, type=int, help='空间下采样因子，2/4/8')
    parser.add_argument('--fpn_out_channels', default=256, type=int, help='FPN输出通道数')

    # 视觉-语义多模态辅助分支：默认关闭，打开后不改变原 R2 主分类头，只加 semantic logits 小权重融合
    parser.add_argument('--use_semantic_branch', default=False, type=U.str2bool,
                        help='启用 R2 + Semantic Class Embedding 视觉-语义辅助分支')
    parser.add_argument('--sem_aux_only', default=False, type=U.str2bool,
                        help='Use semantics only as a training loss; validation and inference logits remain pure R2')
    parser.add_argument('--sem_dim', default=256, type=int,
                        help='Semantic class embedding dimension')
    parser.add_argument('--sem_dropout', default=0.1, type=float,
                        help='Dropout used in semantic visual projection')
    parser.add_argument('--sem_temperature', default=1.0, type=float,
                        help='Cosine similarity temperature for semantic logits')
    parser.add_argument('--sem_gamma_init', default=0.1, type=float,
                        help='Initial fusion weight gamma for semantic logits')
    parser.add_argument('--sem_text_embed_path', default='', type=str,
                        help='Optional .pt file with LLM/text prompt embeddings for semantic branch')
    parser.add_argument('--sem_text_center', default=False, type=U.str2bool,
                        help='Remove the shared language mean from class text prototypes')
    parser.add_argument(
        '--sem_text_transform',
        default='legacy',
        choices=['legacy', 'none', 'mean', 'pc1', 'whiten'],
        help=(
            'Prototype transform. legacy preserves --sem_text_center; explicit '
            'modes enable centering alternatives for ablation.'
        ),
    )
    parser.add_argument('--sem_text_transform_eps', default=1e-5, type=float,
                        help='Numerical epsilon for PC/whitening prototype transforms')
    parser.add_argument('--sem_prompt_trainable', default=True, type=U.str2bool,
                        help='Whether semantic prompt tokens or residual deltas are trainable')
    parser.add_argument('--sem_prompt_residual', default=True, type=U.str2bool,
                        help='Use text embedding + zero-init trainable delta instead of directly fine-tuning text tokens')
    parser.add_argument('--sem_gamma_max', default=0.2, type=float,
                        help='Upper bound for semantic logits fusion weight; <=0 disables bounding')
    parser.add_argument('--sem_gamma_trainable', default=True, type=U.str2bool,
                        help='Whether semantic fusion gamma is trainable; false keeps gamma fixed')
    parser.add_argument('--sem_classwise_gamma', default=False, type=U.str2bool,
                        help='Use a separate bounded semantic fusion gamma for each class')
    parser.add_argument('--sem_use_gate', default=False, type=U.str2bool,
                        help='Use an image-conditioned class-wise gate for semantic logits')
    parser.add_argument('--sem_gate_hidden', default=0, type=int,
                        help='Hidden dimension of semantic gate MLP; 0 uses a conservative default')
    parser.add_argument('--sem_gate_init', default=0.5, type=float,
                        help='Initial sigmoid probability for semantic gate')
    parser.add_argument('--sem_view_calib', default=False, type=U.str2bool,
                        help='Use dual-view semantic reliability to calibrate LLM semantic correction')
    parser.add_argument('--sem_view_calib_min', default=0.7, type=float,
                        help='Minimum multiplier for view-aware semantic calibration')
    parser.add_argument('--sem_view_calib_max', default=1.3, type=float,
                        help='Maximum multiplier for view-aware semantic calibration')
    parser.add_argument(
        '--sem_basc_mode',
        default='none',
        choices=['none', 'norm', 'uncertainty', 'full'],
        help='Backbone-agnostic semantic calibration: none/norm/uncertainty/full',
    )
    parser.add_argument('--sem_basc_compat_scale', default=2.0, type=float,
                        help='Compatibility sharpness used by full BASC')
    parser.add_argument('--sem_basc_eps', default=1e-5, type=float,
                        help='Numerical epsilon for BASC per-sample logit normalization')
    parser.add_argument('--sem_trust_router', default=False, type=U.str2bool,
                        help='Enable class-shared, backbone-agnostic semantic trust routing')
    parser.add_argument('--sem_trust_hidden', default=16, type=int,
                        help='Hidden width of the shared semantic trust router')
    parser.add_argument('--sem_trust_init', default=0.05, type=float,
                        help='Initial semantic trust probability')
    parser.add_argument('--sem_trust_uncertainty_floor', default=0.1, type=float,
                        help='Minimum semantic candidate scale for confident base predictions')
    parser.add_argument('--sem_trust_classwise', default=False, type=U.str2bool,
                        help='Add a learned per-class risk budget to the semantic trust router')
    parser.add_argument('--sem_trust_candidate_mode', default='bounded',
                        choices=['bounded', 'frozen'],
                        help='Use conservative normalized candidate or preserve Frozen-LLM correction')
    parser.add_argument('--sem_trust_lambda', default=0.0, type=float,
                        help='Weight for target-derived semantic trust supervision')
    parser.add_argument('--sem_trust_temperature', default=0.01, type=float,
                        help='Soft trust-target temperature in per-class BCE improvement units')
    parser.add_argument('--sem_trust_margin', default=0.0, type=float,
                        help='Required BCE improvement before a semantic candidate is trusted')
    parser.add_argument('--sem_trust_target_mode', default='soft',
                        choices=['soft', 'asymmetric'],
                        help='Soft benefit target or hard asymmetric semantic-risk target')
    parser.add_argument('--sem_trust_harm_weight', default=3.0, type=float,
                        help='Extra weight for harmful candidates in asymmetric trust supervision')
    parser.add_argument('--sem_rank_lambda', default=0.0, type=float,
                        help='Weight for visual-anchored semantic ranking calibration; 0 disables it')
    parser.add_argument('--sem_rank_guard_weight', default=2.0, type=float,
                        help='Relative weight for preserving base-correct positive/negative rankings')
    parser.add_argument('--sem_rank_temperature', default=0.2, type=float,
                        help='Temperature for semantic positive/negative ranking')
    parser.add_argument('--sem_rank_need_temperature', default=0.2, type=float,
                        help='Temperature for weighting pairs by base-branch ranking need')
    parser.add_argument('--sem_conflict_lambda', default=0.0, type=float,
                        help='Weight for target-aware semantic conflict suppression loss; 0 disables it')
    parser.add_argument('--sem_conflict_base_margin', default=0.0, type=float,
                        help='Base-branch label-alignment margin for semantic conflict suppression')
    parser.add_argument('--sem_conflict_sem_margin', default=0.0, type=float,
                        help='Semantic correction margin for semantic conflict suppression')
    parser.add_argument('--sem_freeze_base', default=False, type=U.str2bool,
                        help='Freeze the loaded visual R2 path and train only semantic adapters')
    parser.add_argument('--sem_error_lambda', default=0.0, type=float,
                        help='Weight for visual-error-guided semantic residual supervision')
    parser.add_argument('--sem_error_power', default=2.0, type=float,
                        help='Power used to focus semantic supervision on base errors')
    parser.add_argument('--sem_error_min_weight', default=0.05, type=float,
                        help='Minimum per-label weight in visual-error semantic supervision')
    parser.add_argument('--sem_error_guard_weight', default=4.0, type=float,
                        help='Penalty for semantic corrections opposing reliable base predictions')

    # Spatial attribute query branch. It is independent of the legacy global
    # semantic branch and remains fully disabled unless explicitly requested.
    parser.add_argument('--use_spatial_query', default=False, type=U.str2bool,
                        help='Enable class-attribute queries over unpooled dual-view C3/C4 features')
    parser.add_argument('--spatial_query_source', default='random',
                        choices=['random', 'llm'],
                        help='Initialize isomorphic attribute queries from random or LLM text embeddings')
    parser.add_argument('--spatial_attribute_embed_path', default='', type=str,
                        help='[C,K,D] attribute embedding file required by spatial_query_source=llm')
    parser.add_argument('--spatial_query_dim', default=256, type=int)
    parser.add_argument('--spatial_query_text_dim', default=384, type=int)
    parser.add_argument('--spatial_query_attributes', default=4, type=int)
    parser.add_argument('--spatial_query_heads', default=4, type=int)
    parser.add_argument('--spatial_query_dropout', default=0.1, type=float)
    parser.add_argument('--spatial_query_c3_size', default=14, type=int)
    parser.add_argument('--spatial_query_c4_size', default=7, type=int)
    parser.add_argument('--spatial_query_gamma_init', default=0.05, type=float)
    parser.add_argument('--spatial_query_gamma_max', default=0.2, type=float)
    parser.add_argument('--spatial_query_uncertainty_floor', default=0.1, type=float)
    parser.add_argument('--spatial_query_view_temperature', default=0.2, type=float)
    parser.add_argument('--spatial_query_random_seed', default=-1, type=int,
                        help='Random-query seed; negative reuses the experiment seed')
    parser.add_argument('--spatial_query_freeze_base', default=False, type=U.str2bool,
                        help='Freeze loaded R2 and train only the spatial query head')
    parser.add_argument('--spatial_query_lambda', default=0.0, type=float,
                        help='Weight of class-balanced spatial-query supervision')
    parser.add_argument('--spatial_query_final_weight', default=0.25, type=float,
                        help='Extra balanced supervision weight on corrected final logits')
    parser.add_argument('--spatial_query_guard_weight', default=1.0, type=float,
                        help='Penalty for corrections opposing reliable R2 predictions')
    parser.add_argument('--spatial_query_negative_weight', default=1.0, type=float,
                        help='Negative term weight in class-balanced query BCE')

    parser.add_argument('--use_view_evidence_distill', default=False, type=U.str2bool,
                        help='Enable training-only cross-view best-evidence distillation')
    parser.add_argument('--view_evidence_projection_dim', default=128, type=int,
                        help='Per-level projection width of the shared single-view head')
    parser.add_argument('--view_evidence_hidden_dim', default=256, type=int,
                        help='Hidden width of the shared single-view classifier')
    parser.add_argument('--view_evidence_dropout', default=0.1, type=float)
    parser.add_argument('--view_evidence_aux_weight', default=0.0, type=float,
                        help='Weight of shared A/B single-view deep supervision')
    parser.add_argument('--view_evidence_distill_weight', default=0.0, type=float,
                        help='Weight of label-guided best-view evidence distillation')
    parser.add_argument('--view_evidence_warmup_epochs', default=5, type=int,
                        help='Train view heads for this many epochs before distillation')
    parser.add_argument('--view_evidence_ramp_epochs', default=5, type=int,
                        help='Linear ramp length for best-evidence distillation')
    parser.add_argument('--view_evidence_advantage_temperature', default=0.1, type=float,
                        help='Scale mapping teacher BCE advantage to distillation confidence')
    parser.add_argument('--view_evidence_negative_weight', default=1.0, type=float,
                        help='Negative term weight in balanced single-view supervision')
    parser.add_argument('--use_selective_view_rescue', default=False, type=U.str2bool)
    parser.add_argument('--selective_rescue_projection_dim', default=128, type=int)
    parser.add_argument('--selective_rescue_hidden_dim', default=256, type=int)
    parser.add_argument('--selective_rescue_dropout', default=0.1, type=float)
    parser.add_argument('--selective_rescue_gamma_max', default=0.05, type=float)
    parser.add_argument('--selective_rescue_uncertainty_threshold', default=0.5, type=float)
    parser.add_argument('--selective_rescue_gate_temperature', default=0.1, type=float)
    parser.add_argument('--selective_rescue_detach_features', default=True, type=U.str2bool)
    parser.add_argument('--selective_rescue_aux_weight', default=0.0, type=float)
    parser.add_argument('--selective_rescue_loss_weight', default=0.0, type=float)
    parser.add_argument('--selective_rescue_guard_weight', default=0.0, type=float)
    parser.add_argument('--selective_rescue_warmup_epochs', default=5, type=int)
    parser.add_argument('--selective_rescue_ramp_epochs', default=5, type=int)
    parser.add_argument('--selective_rescue_aux_decay_start', default=30, type=int)
    parser.add_argument('--selective_rescue_aux_decay_end', default=80, type=int)
    parser.add_argument('--use_frozen_anchor_rescue', default=False, type=U.str2bool)
    parser.add_argument('--frozen_rescue_aux_only', default=False, type=U.str2bool)
    parser.add_argument('--frozen_rescue_freeze_base', default=False, type=U.str2bool)
    parser.add_argument('--frozen_rescue_projection_dim', default=128, type=int)
    parser.add_argument('--frozen_rescue_hidden_dim', default=256, type=int)
    parser.add_argument('--frozen_rescue_dropout', default=0.0, type=float)
    parser.add_argument('--frozen_rescue_gamma_max', default=0.03, type=float)
    parser.add_argument('--frozen_rescue_uncertainty_threshold', default=0.35, type=float)
    parser.add_argument('--frozen_rescue_gate_temperature', default=0.1, type=float)
    parser.add_argument('--frozen_rescue_trust_init', default=0.05, type=float)
    parser.add_argument('--frozen_rescue_aux_weight', default=0.0, type=float)
    parser.add_argument('--frozen_rescue_trust_weight', default=0.0, type=float)
    parser.add_argument('--frozen_rescue_rank_weight', default=0.0, type=float)
    parser.add_argument('--frozen_rescue_guard_weight', default=0.0, type=float)
    parser.add_argument('--frozen_rescue_rank_temperature', default=0.2, type=float)
    parser.add_argument('--use_frozen_region_rescue', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_aux_only', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_freeze_base', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_levels', default=['C3', 'C4'], nargs='+')
    parser.add_argument('--frozen_region_projection_dim', default=64, type=int)
    parser.add_argument('--frozen_region_topk_ratio', default=0.1, type=float)
    parser.add_argument('--frozen_region_temperature', default=0.2, type=float)
    parser.add_argument('--frozen_region_gamma_init', default=0.003, type=float)
    parser.add_argument('--frozen_region_gamma_max', default=0.02, type=float)
    parser.add_argument('--frozen_region_uncertainty_threshold', default=0.35, type=float)
    parser.add_argument('--frozen_region_gate_temperature', default=0.1, type=float)
    parser.add_argument('--frozen_region_trust_init', default=0.05, type=float)
    parser.add_argument('--frozen_region_aux_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_trust_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_rank_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_guard_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_rank_temperature', default=0.1, type=float)
    parser.add_argument('--frozen_region_target_gain', default=0.005, type=float)
    parser.add_argument('--use_frozen_counterfactual_router', default=False, type=U.str2bool)
    parser.add_argument('--frozen_counterfactual_aux_only', default=False, type=U.str2bool)
    parser.add_argument('--frozen_counterfactual_freeze_base', default=False, type=U.str2bool)
    parser.add_argument('--frozen_counterfactual_hidden_dim', default=32, type=int)
    parser.add_argument('--frozen_counterfactual_class_embed_dim', default=8, type=int)
    parser.add_argument('--frozen_counterfactual_rho_init', default=0.05, type=float)
    parser.add_argument('--frozen_counterfactual_rho_max', default=0.2, type=float)
    parser.add_argument('--frozen_counterfactual_rescue_init', default=0.1, type=float)
    parser.add_argument('--frozen_counterfactual_delta_clip', default=6.0, type=float)
    parser.add_argument('--frozen_counterfactual_route_weight', default=0.0, type=float)
    parser.add_argument('--frozen_counterfactual_rank_weight', default=0.0, type=float)
    parser.add_argument('--frozen_counterfactual_guard_weight', default=0.0, type=float)
    parser.add_argument('--frozen_counterfactual_residual_weight', default=0.0, type=float)
    parser.add_argument('--frozen_counterfactual_router_temperature', default=0.02, type=float)
    parser.add_argument('--frozen_counterfactual_router_margin', default=0.02, type=float)
    parser.add_argument('--frozen_counterfactual_rank_temperature', default=0.2, type=float)
    parser.add_argument('--frozen_counterfactual_rank_margin', default=0.5, type=float)
    parser.add_argument('--frozen_counterfactual_hard_threshold', default=2.0, type=float)
    parser.add_argument('--frozen_counterfactual_guard_threshold', default=4.0, type=float)
    parser.add_argument('--frozen_counterfactual_queue_size', default=128, type=int)
    parser.add_argument('--frozen_counterfactual_region_level', default='', type=str)
    parser.add_argument('--frozen_counterfactual_region_projection_dim', default=32, type=int)
    parser.add_argument('--frozen_counterfactual_region_temperature', default=0.2, type=float)
    # M9: frozen-anchor region interaction mixture of experts. Every option is
    # disabled by default so existing methods keep their original behavior.
    parser.add_argument('--use_frozen_region_interaction_moe', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_interaction_aux_only', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_interaction_freeze_base', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_interaction_router_only', default=False, type=U.str2bool)
    parser.add_argument('--frozen_region_interaction_levels', default=['C3', 'C4'], nargs='+')
    parser.add_argument('--frozen_region_interaction_projection_dim', default=64, type=int)
    parser.add_argument('--frozen_region_interaction_temperature', default=0.2, type=float)
    parser.add_argument('--frozen_region_interaction_shared_axis', default='width', choices=['width', 'height'])
    parser.add_argument('--frozen_region_interaction_axis_radius', default=1, type=int)
    parser.add_argument('--frozen_region_interaction_residual_init', default=0.03, type=float)
    parser.add_argument('--frozen_region_interaction_residual_max', default=0.2, type=float)
    parser.add_argument('--frozen_region_interaction_router_hidden_dim', default=128, type=int)
    parser.add_argument('--frozen_region_interaction_class_embed_dim', default=16, type=int)
    parser.add_argument('--frozen_region_interaction_rho_init', default=0.08, type=float)
    parser.add_argument('--frozen_region_interaction_rho_max', default=0.25, type=float)
    parser.add_argument('--frozen_region_interaction_rescue_init', default=0.05, type=float)
    parser.add_argument('--frozen_region_interaction_delta_clip', default=6.0, type=float)
    parser.add_argument('--frozen_region_interaction_expert_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_alignment_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_diversity_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_route_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_rank_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_guard_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_residual_weight', default=0.0, type=float)
    parser.add_argument('--frozen_region_interaction_expert_temperature', default=0.1, type=float)
    parser.add_argument('--frozen_region_interaction_router_temperature', default=0.03, type=float)
    parser.add_argument('--frozen_region_interaction_router_margin', default=0.01, type=float)
    parser.add_argument('--frozen_region_interaction_rank_temperature', default=0.2, type=float)
    parser.add_argument('--frozen_region_interaction_rank_margin', default=0.5, type=float)
    parser.add_argument('--frozen_region_interaction_hard_threshold', default=2.0, type=float)
    parser.add_argument('--frozen_region_interaction_guard_threshold', default=4.0, type=float)
    parser.add_argument('--frozen_region_interaction_queue_size', default=128, type=int)
    parser.add_argument('--frozen_region_interaction_include_counterfactual', default=False, type=U.str2bool)

    parser.add_argument('--use_dvcre', default=False, type=U.str2bool,
                        help='Enable dual-view class-wise region complement fusion')
    parser.add_argument('--dvcre_levels', default=['C3', 'C4'], nargs='+')
    parser.add_argument('--dvcre_projection_dim', default=64, type=int)
    parser.add_argument('--dvcre_topk_ratio', default=0.25, type=float)
    parser.add_argument('--dvcre_temperature', default=0.2, type=float)
    parser.add_argument('--dvcre_residual_init', default=0.05, type=float)
    parser.add_argument('--dvcre_residual_max', default=0.2, type=float)
    parser.add_argument('--dvcre_aux_weight', default=0.0, type=float,
                        help='Weight of class-balanced DV-CRE region supervision')

    parser.add_argument('--use_iscvf', default=False, type=U.str2bool,
                        help='Enable intervention-stable cross-view fusion')
    parser.add_argument('--iscvf_levels', default=['C3', 'C4', 'C5'], nargs='+')
    parser.add_argument('--iscvf_gate_reduction', default=16, type=int)
    parser.add_argument('--iscvf_keep_prob', default=0.75, type=float)
    parser.add_argument('--iscvf_intervention_weight', default=0.0, type=float,
                        help='Weight of supervised intervention-view training')
    parser.add_argument('--iscvf_consistency_weight', default=0.0, type=float,
                        help='Weight of confidence-aware intervention consistency')
    parser.add_argument('--iscvf_warmup_epochs', default=5, type=int)
    parser.add_argument('--iscvf_ramp_epochs', default=5, type=int)

    parser.add_argument('--use_visual_evidence_router', default=False, type=U.str2bool,
                        help='Enable visual-only sample/class/region evidence routing')
    parser.add_argument('--visual_route_mode', default='pg_cver',
                        choices=['pg_cver', 'sa_dca', 'ca_rer', 'cycle_cver'])
    parser.add_argument('--visual_route_level', default='C4', choices=['C3', 'C4', 'C5'])
    parser.add_argument('--visual_route_projection_dim', default=64, type=int)
    parser.add_argument('--visual_route_topk_ratio', default=0.25, type=float)
    parser.add_argument('--visual_route_temperature', default=0.2, type=float)
    parser.add_argument('--visual_route_shared_axis', default='width',
                        choices=['width', 'height'])
    parser.add_argument('--visual_route_axis_radius', default=1, type=int)
    parser.add_argument('--visual_route_gate_init', default=0.05, type=float)
    parser.add_argument('--visual_route_gamma_init', default=0.05, type=float)
    parser.add_argument('--visual_route_gamma_max', default=0.25, type=float)
    parser.add_argument('--visual_route_reject_temperature', default=0.1, type=float)
    parser.add_argument('--visual_route_dropout', default=0.1, type=float)
    parser.add_argument('--visual_route_aux_weight', default=0.0, type=float,
                        help='Weight for class-conditioned region evidence supervision')
    parser.add_argument('--visual_route_rescue_weight', default=0.0, type=float,
                        help='Weight for error-focused corrected-logit supervision')
    parser.add_argument('--visual_route_guard_weight', default=0.0, type=float,
                        help='Weight penalizing corrections against reliable R2 evidence')
    parser.add_argument('--visual_route_cycle_weight', default=0.0, type=float,
                        help='Cycle consistency weight; used only by cycle_cver')
    parser.add_argument('--visual_route_negative_weight', default=1.0, type=float)

    parser.add_argument('--use_p9_caprs', default=False, type=U.str2bool)
    parser.add_argument('--use_p10_wgcr', default=False, type=U.str2bool)
    parser.add_argument('--use_p11_otcvr', default=False, type=U.str2bool)
    parser.add_argument('--use_p12_berf', default=False, type=U.str2bool)
    parser.add_argument('--use_p13_vdrm', default=False, type=U.str2bool)
    parser.add_argument('--use_p14_hcaer', default=False, type=U.str2bool)
    parser.add_argument('--use_p15_vtr', default=False, type=U.str2bool)
    parser.add_argument('--use_p16_facgr', default=False, type=U.str2bool)
    parser.add_argument('--use_p17_dcasr', default=False, type=U.str2bool)
    parser.add_argument('--use_p18_ewsar', default=False, type=U.str2bool)
    parser.add_argument('--use_p19_apcer', default=False, type=U.str2bool)
    parser.add_argument('--use_p20_cvcr', default=False, type=U.str2bool)
    parser.add_argument(
        '--plain_innovation_freeze_base', default=False, type=U.str2bool,
        help='Freeze a loaded Plain-BCE anchor and train only the innovation head',
    )
    parser.add_argument(
        '--plain_innovation_levels', default=['C4', 'C5'], nargs='+'
    )
    parser.add_argument('--plain_innovation_projection_dim', default=64, type=int)
    parser.add_argument('--plain_innovation_topk', default=8, type=int)
    parser.add_argument('--plain_innovation_region_pooling', default='topk', choices=('topk', 'gap'))
    parser.add_argument('--plain_innovation_temperature', default=0.2, type=float)
    parser.add_argument('--plain_innovation_dropout', default=0.1, type=float)
    parser.add_argument('--plain_innovation_gamma_init', default=0.005, type=float)
    parser.add_argument('--plain_innovation_gamma_max', default=0.05, type=float)
    parser.add_argument('--plain_innovation_base_floor', default=0.8, type=float)
    parser.add_argument(
        '--plain_innovation_use_counterfactual_experts',
        default=True,
        type=U.str2bool,
        help='P9 ablation: include the OL-only and SD-only routing experts.',
    )
    parser.add_argument(
        '--plain_innovation_use_learned_router',
        default=True,
        type=U.str2bool,
        help='P9 ablation: learn class-wise expert weights; false uses uniform weights.',
    )
    parser.add_argument(
        '--plain_innovation_router_variant',
        default='legacy',
        choices=('legacy', 'residual_nonbase', 'selective_regret'),
        help=(
            'P9 router parameterization. residual_nonbase keeps the base '
            'outside a softmax over the three non-base experts; '
            'selective_regret learns three direction-aware utilities.'
        ),
    )
    parser.add_argument(
        '--plain_innovation_gamma_trainable',
        default=True,
        type=U.str2bool,
        help='If false, P9 uses gamma_init as an exact fixed residual budget.',
    )
    parser.add_argument(
        '--plain_innovation_router_only',
        default=False,
        type=U.str2bool,
        help='Freeze the loaded base and P9 experts; train only the P9 router.',
    )
    parser.add_argument(
        '--plain_innovation_reset_router_after_finetune',
        default=False,
        type=U.str2bool,
        help='Reset the P9 router after loading finetune weights and before freezing.',
    )
    parser.add_argument(
        '--plain_innovation_transfer_legacy_router',
        default=False,
        type=U.str2bool,
        help=(
            'For selective_regret, transfer legacy router output rows 1:4 '
            'after loading a Final checkpoint.'
        ),
    )
    parser.add_argument('--plain_innovation_warmup_epochs', default=15, type=int)
    parser.add_argument('--plain_innovation_ramp_epochs', default=10, type=int)
    parser.add_argument('--plain_innovation_sinkhorn_iters', default=4, type=int)
    parser.add_argument('--plain_innovation_router_start_epoch', default=8, type=int)
    parser.add_argument('--plain_innovation_router_ramp_epochs', default=3, type=int)
    parser.add_argument('--plain_innovation_gate_init', default=0.15, type=float)
    parser.add_argument('--plain_innovation_trust_threshold', default=0.5, type=float)
    parser.add_argument('--plain_innovation_trust_temperature', default=0.1, type=float)
    parser.add_argument('--plain_innovation_uncertainty_floor', default=0.25, type=float)
    parser.add_argument('--plain_innovation_budget_target', default=0.12, type=float)
    parser.add_argument(
        '--plain_innovation_advantage_temperature', default=0.05, type=float
    )
    parser.add_argument(
        '--plain_innovation_gain_margin', default=1e-5, type=float
    )
    parser.add_argument(
        '--plain_innovation_score_temperature', default=1.0, type=float
    )
    parser.add_argument(
        '--plain_innovation_corrupt_probability', default=0.75, type=float
    )
    parser.add_argument(
        '--plain_innovation_corrupt_ratio', default=0.25, type=float
    )
    parser.add_argument(
        '--plain_innovation_full_view_drop_probability',
        default=0.35,
        type=float,
    )
    parser.add_argument(
        '--plain_innovation_complement_margin', default=0.02, type=float
    )
    parser.add_argument(
        '--plain_innovation_complement_temperature', default=0.05, type=float
    )
    parser.add_argument('--plain_innovation_aux_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_route_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_guard_weight', default=0.0, type=float)
    parser.add_argument(
        '--plain_innovation_consistency_weight', default=0.0, type=float
    )
    parser.add_argument('--plain_innovation_regret_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_evidence_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_match_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_single_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_rank_weight', default=0.0, type=float)
    parser.add_argument('--plain_innovation_budget_weight', default=0.0, type=float)

    parser.add_argument('--patience', type=int, default=0,
                        help='Enable early stopping if validation metric does not improve for this many epochs. Default 0 to disable.')
    parser.add_argument('--early_stop_start_epoch', type=int, default=0,
                        help='First epoch counted by early stopping; default 0 preserves existing behavior.')
    parser.add_argument('--ahcr_mode', default='intra_level', choices=['intra_level', 'inter_level'],
                        help="Defines the hierarchical strategy for AHCR fusion. Only used if fuse_mode is 'ahcr'.")

    parser.add_argument('--teacher_mode', default=True, type=U.str2bool)
    parser.add_argument('--ema_decay', default=0.9999, type=float)
    parser.add_argument('--ema_device', default='cpu')
    parser.add_argument('--fsdp_cpu_offload', default=False, type=U.str2bool)

    parser.add_argument('--resume', default='', help='从检查点恢复训练 (checkpoint_last.pth 或 checkpoint_best.pth)')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='从指定epoch开始（默认自动检测）')
    parser.add_argument('--resume_optimizer', default=True, type=U.str2bool, help='是否恢复优化器状态')
    parser.add_argument('--resume_scheduler', default=True, type=U.str2bool, help='是否恢复学习率调度器')

    parser.add_argument('--use_distillation', type=U.str2bool, default=False)
    parser.add_argument('--teacher_model', type=str, default='convnext_small')
    parser.add_argument('--teacher_weights', type=str, default='')
    parser.add_argument('--kd_mode', type=str, default='logits', choices=['logits', 'dkd'])
    parser.add_argument('--distillation_alpha', type=float, default=0.5, help="Hard loss weight.")
    parser.add_argument('--distillation_tau', type=float, default=2.0)
    parser.add_argument('--distill_feature_layers', type=str, nargs='+', default=None)
    parser.add_argument('--distillation_beta', type=float, default=0.0, help="Feature loss weight.")
    parser.add_argument('--dkd_alpha', type=float, default=1.0)
    parser.add_argument('--dkd_beta', type=float, default=8.0)

    parser.add_argument('--base_loss', type=str, default='bce',
                        choices=['bce', 'mlsm', 'focal', 'asl', 'fals', 'mcb', 'gebce', 'dals', 'mcb_convex'],
                        help='选择基础监督损失：bce / mlsm / focal / asl / fals / mcb / gebce / dals / mcb_convex')

    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=None)
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    parser.add_argument('--asl_clip', type=float, default=0.05)

    parser.add_argument('--fals_eps', type=float, default=0.1)
    parser.add_argument('--fals_gamma', type=float, default=2.0)

    parser.add_argument('--mcb_momentum', type=float, default=0.9)

    parser.add_argument('--ge_lambda', type=float, default=0.1,
                        help='GE-BCE: class-level gradient equalization strength')
    parser.add_argument('--ge_pos_only', type=U.str2bool, default=True,
                        help='GE-BCE: use positives only to compute G_c')
    parser.add_argument('--ge_alpha', type=float, default=0.75,
                        help='GE-BCE: weight for positives when pos_only is False')
    parser.add_argument('--ge_ema', type=U.str2bool, default=True,
                        help='GE-BCE: EMA smoothing over G_c')
    parser.add_argument('--ge_momentum', type=float, default=0.9,
                        help='GE-BCE: EMA momentum')
    parser.add_argument('--ge_band', type=float, default=0.0,
                        help='GE-BCE: tolerance band; diffs within band are not penalized')

    parser.add_argument('--dals_eps', type=float, default=0.1)
    parser.add_argument('--dals_gamma', type=float, default=2.0)

    parser.add_argument('--mcb_tau', type=float, default=1.0)
    parser.add_argument('--mcb_wmin', type=float, default=1e-3)

    parser.add_argument('--ge_trainable', type=U.str2bool, default=False,
                        help='GE 正则是否参与反传（true=非凸；false=仅诊断）')

    parser.add_argument('--aug_mode', default='standard',
                        choices=['none', 'standard', 'conditional', 'conditional_4', 'rand_aug', 'trivial_aug'],
                        help="Select the data augmentation strategy.")
    parser.add_argument('--rand_aug_n', type=int, default=2, help="Hyperparameter N for RandAugment.")
    parser.add_argument('--rand_aug_m', type=int, default=9, help="Hyperparameter M for RandAugment (0-30).")

    parser.add_argument('--eval_csv', default='')
    parser.add_argument('--summary_csv', default='viewaware_llm_5seed_detailed.csv', type=str,
                        help='CSV file receiving the final per-class experiment row')
    parser.add_argument('--deterministic', default=False, type=U.str2bool,
                        help='Use deterministic CUDA behavior for paired reproducibility experiments')
    parser.add_argument('--reseed_before_training', default=False, type=U.str2bool,
                        help='Reset RNG after model construction for paired architecture comparisons')
    parser.add_argument('--eval_csv_sort_classes', default=False, type=U.str2bool,
                        help="If true, sort per-class AP columns by descending AP when writing CSV.")
    
    parser.add_argument('--gspf_lambda_consistency', type=float, default=0.0,
                        help='GSPF: multi-level prototype usage consistency loss weight')
    parser.add_argument('--gspf_lambda_ortho', type=float, default=0.0,
                        help='GSPF: prototype orthogonality/diversity loss weight')
    parser.add_argument('--eval_csv_topk', default=0, type=int,
                        help="If >0, write per-sample top-k predictions (class:score) to a separate CSV per epoch.")
    
    parser.add_argument('--cv_lambda_sem', type=float, default=0.0,
                        help='CV-GSC: cross-view semantic consistency loss weight')
    parser.add_argument('--cv_lambda_geo', type=float, default=0.0,
                        help='CV-GSC: cross-view geometry response consistency loss weight')
    parser.add_argument('--use_cv_gsc', default=False, type=U.str2bool,
                        help='是否启用 CV-GSC 跨视角几何-语义一致性模块')
    parser.add_argument('--cv_spatial_reduction', type=int, default=4,
                        help='CV-GSC 空间降采样倍率，建议 4/8')
    return parser


def build_model(args):
    if args.model == "dagnet_official_adapter":
        from models.fair_baseline_adapters import DAGNetAdapter

        if int(args.input_size) != DAGNetAdapter.required_input_size:
            raise ValueError("official DAGNet architecture requires --input_size 256")
        return DAGNetAdapter(
            num_classes=args.num_classes,
            source_dir="third_party/DAGNet_official",
            source_commit="ab4e3ff202af5328eedb61c8953c4069e0bf8fee",
        )

    if args.model == "resnet50_ml_decoder_adapter":
        from models.fair_baseline_adapters import MLDecoderDualViewAdapter

        return MLDecoderDualViewAdapter(
            num_classes=args.num_classes,
            source_dir="third_party/ML_Decoder_official",
            source_commit="8a9e984f671c9c30c98d2c45dfcaf4383381c254",
        )

    if args.model == "official_ahcr_resnet50":
        from models.official_ahcr_adapter import OfficialAHCRAdapter

        print(
            "Official AHCR adapter: "
            f"commit={getattr(args, 'official_ahcr_source_commit', '')} "
            f"weights={getattr(args, 'official_ahcr_pretrained_weights', 'IMAGENET1K_V2')}"
        )
        return OfficialAHCRAdapter(
            num_classes=args.num_classes,
            source_dir=getattr(
                args, "official_ahcr_source_dir", "third_party/DvXray_official"
            ),
            source_commit=getattr(
                args,
                "official_ahcr_source_commit",
                "a6bfc1b1299d28e8226c106a94967287a8e30927",
            ),
            pretrained_weights=getattr(
                args, "official_ahcr_pretrained_weights", "IMAGENET1K_V2"
            ),
        )

    backbone_builder = None
    if (tv_backbones is not None) and hasattr(tv_backbones, args.model):
        backbone_builder = getattr(tv_backbones, args.model)
        print(f"✅ 从 [tv_backbones] 找到模型构建器: {args.model}")

    elif (timm_backbones is not None) and hasattr(timm_backbones, args.model):
        backbone_builder = getattr(timm_backbones, args.model)
        print(f"✅ 从 [timm_backbones] 找到模型构建器: {args.model}")

    elif hasattr(convnextv2, args.model):
        backbone_builder = getattr(convnextv2, args.model)
        print(f"✅ 从 [ConvNeXtV2] 模块中成功找到模型构建器: {args.model}")

    elif hasattr(convnextv1, args.model):
        backbone_builder = getattr(convnextv1, args.model)
        print(f"✅ 从 [ConvNeXtV1] 模块中成功找到模型构建器: {args.model}")

    else:
        raise ValueError(f"未找到模型构建器: {args.model}")

    try:
        backbone = backbone_builder(num_classes=0)
    except TypeError:
        backbone = backbone_builder()

    sig = inspect.signature(ConvNeXtV2Dual.__init__)
    valid_keys = set(sig.parameters.keys())

    out_idx = tuple(getattr(args, "out_indices", (1, 2, 3)))
    fuse_kw = getattr(args, "fuse_mode", "add")
    base_head_type = getattr(args, "head_type", "c5")

    parsed_attention_config = None
    if getattr(args, "attention_name", None):
        parsed_attention_config = resolve_attention_config(args.attention_name)
        print(f"✅ 使用 attention_name={args.attention_name}")
        print("✅ 映射后的注意力配置:", parsed_attention_config)
    elif getattr(args, "attention_config", None):
        try:
            parsed_attention_config = json.loads(args.attention_config)
            print("⚠️ 使用 legacy --attention_config")
            print("✅ 成功解析注意力配置:", parsed_attention_config)
        except json.JSONDecodeError:
            raise ValueError(f"错误: 解析 --attention_config 的 JSON 字符串失败: {args.attention_config}")

    candidate_kwargs = {
        "backbone": backbone,
        "num_classes": args.num_classes,
        "fuse_mode": fuse_kw,
        "return_intermediate": getattr(args, "return_intermediate", False),
        "out_indices": out_idx,
        "fuse_levels": getattr(args, "fuse_levels", None),
        "head_type": base_head_type,
        "xattn_heads": getattr(args, "xattn_heads", 4),
        "xattn_reduction": getattr(args, "xattn_reduction", 4),
        "fpn_out_channels": getattr(args, "fpn_out_channels", 256),
        "attention_config": parsed_attention_config,
        "ahcr_mode": getattr(args, "ahcr_mode", "intra_level"),
        "use_cv_gsc": getattr(args, "use_cv_gsc", False),
        "cv_spatial_reduction": getattr(args, "cv_spatial_reduction", 4),
        "use_semantic_branch": getattr(args, "use_semantic_branch", False),
        "sem_aux_only": getattr(args, "sem_aux_only", False),
        "sem_dim": getattr(args, "sem_dim", 256),
        "sem_dropout": getattr(args, "sem_dropout", 0.1),
        "sem_temperature": getattr(args, "sem_temperature", 1.0),
        "sem_gamma_init": getattr(args, "sem_gamma_init", 0.1),
        "sem_text_embed_path": getattr(args, "sem_text_embed_path", "") or None,
        "sem_text_center": getattr(args, "sem_text_center", False),
        "sem_text_transform": getattr(args, "sem_text_transform", "legacy"),
        "sem_text_transform_eps": getattr(args, "sem_text_transform_eps", 1e-5),
        "sem_prompt_trainable": getattr(args, "sem_prompt_trainable", True),
        "sem_prompt_residual": getattr(args, "sem_prompt_residual", True),
        "sem_gamma_max": getattr(args, "sem_gamma_max", 0.2),
        "sem_gamma_trainable": getattr(args, "sem_gamma_trainable", True),
        "sem_classwise_gamma": getattr(args, "sem_classwise_gamma", False),
        "sem_use_gate": getattr(args, "sem_use_gate", False),
        "sem_gate_hidden": getattr(args, "sem_gate_hidden", 0),
        "sem_gate_init": getattr(args, "sem_gate_init", 0.5),
        "sem_view_calib": getattr(args, "sem_view_calib", False),
        "sem_view_calib_min": getattr(args, "sem_view_calib_min", 0.7),
        "sem_view_calib_max": getattr(args, "sem_view_calib_max", 1.3),
        "sem_basc_mode": getattr(args, "sem_basc_mode", "none"),
        "sem_basc_compat_scale": getattr(args, "sem_basc_compat_scale", 2.0),
        "sem_basc_eps": getattr(args, "sem_basc_eps", 1e-5),
        "sem_trust_router": getattr(args, "sem_trust_router", False),
        "sem_trust_hidden": getattr(args, "sem_trust_hidden", 16),
        "sem_trust_init": getattr(args, "sem_trust_init", 0.05),
        "sem_trust_uncertainty_floor": getattr(args, "sem_trust_uncertainty_floor", 0.1),
        "sem_trust_classwise": getattr(args, "sem_trust_classwise", False),
        "sem_trust_candidate_mode": getattr(args, "sem_trust_candidate_mode", "bounded"),
        "sem_rank_calibration": getattr(args, "sem_rank_lambda", 0.0) > 0,
        "sem_error_calibration": getattr(args, "sem_error_lambda", 0.0) > 0,
        "use_spatial_query": getattr(args, "use_spatial_query", False),
        "spatial_query_source": getattr(args, "spatial_query_source", "random"),
        "spatial_attribute_embed_path": (
            getattr(args, "spatial_attribute_embed_path", "") or None
        ),
        "spatial_query_dim": getattr(args, "spatial_query_dim", 256),
        "spatial_query_text_dim": getattr(args, "spatial_query_text_dim", 384),
        "spatial_query_attributes": getattr(args, "spatial_query_attributes", 4),
        "spatial_query_heads": getattr(args, "spatial_query_heads", 4),
        "spatial_query_dropout": getattr(args, "spatial_query_dropout", 0.1),
        "spatial_query_c3_size": getattr(args, "spatial_query_c3_size", 14),
        "spatial_query_c4_size": getattr(args, "spatial_query_c4_size", 7),
        "spatial_query_gamma_init": getattr(args, "spatial_query_gamma_init", 0.05),
        "spatial_query_gamma_max": getattr(args, "spatial_query_gamma_max", 0.2),
        "spatial_query_uncertainty_floor": getattr(
            args, "spatial_query_uncertainty_floor", 0.1
        ),
        "spatial_query_view_temperature": getattr(
            args, "spatial_query_view_temperature", 0.2
        ),
        "spatial_query_random_seed": (
            args.seed
            if getattr(args, "spatial_query_random_seed", -1) < 0
            else args.spatial_query_random_seed
        ),
        "use_view_evidence_distill": getattr(
            args, "use_view_evidence_distill", False
        ),
        "view_evidence_projection_dim": getattr(
            args, "view_evidence_projection_dim", 128
        ),
        "view_evidence_hidden_dim": getattr(
            args, "view_evidence_hidden_dim", 256
        ),
        "view_evidence_dropout": getattr(args, "view_evidence_dropout", 0.1),
        "use_selective_view_rescue": getattr(args, "use_selective_view_rescue", False),
        "selective_rescue_projection_dim": getattr(args, "selective_rescue_projection_dim", 128),
        "selective_rescue_hidden_dim": getattr(args, "selective_rescue_hidden_dim", 256),
        "selective_rescue_dropout": getattr(args, "selective_rescue_dropout", 0.1),
        "selective_rescue_gamma_max": getattr(args, "selective_rescue_gamma_max", 0.05),
        "selective_rescue_uncertainty_threshold": getattr(args, "selective_rescue_uncertainty_threshold", 0.5),
        "selective_rescue_gate_temperature": getattr(args, "selective_rescue_gate_temperature", 0.1),
        "selective_rescue_detach_features": getattr(args, "selective_rescue_detach_features", True),
        "use_frozen_anchor_rescue": getattr(args, "use_frozen_anchor_rescue", False),
        "frozen_rescue_aux_only": getattr(args, "frozen_rescue_aux_only", False),
        "frozen_rescue_projection_dim": getattr(args, "frozen_rescue_projection_dim", 128),
        "frozen_rescue_hidden_dim": getattr(args, "frozen_rescue_hidden_dim", 256),
        "frozen_rescue_dropout": getattr(args, "frozen_rescue_dropout", 0.0),
        "frozen_rescue_gamma_max": getattr(args, "frozen_rescue_gamma_max", 0.03),
        "frozen_rescue_uncertainty_threshold": getattr(args, "frozen_rescue_uncertainty_threshold", 0.35),
        "frozen_rescue_gate_temperature": getattr(args, "frozen_rescue_gate_temperature", 0.1),
        "frozen_rescue_trust_init": getattr(args, "frozen_rescue_trust_init", 0.05),
        "use_frozen_region_rescue": getattr(args, "use_frozen_region_rescue", False),
        "frozen_region_aux_only": getattr(args, "frozen_region_aux_only", False),
        "frozen_region_levels": tuple(getattr(args, "frozen_region_levels", ("C3", "C4"))),
        "frozen_region_projection_dim": getattr(args, "frozen_region_projection_dim", 64),
        "frozen_region_topk_ratio": getattr(args, "frozen_region_topk_ratio", 0.1),
        "frozen_region_temperature": getattr(args, "frozen_region_temperature", 0.2),
        "frozen_region_gamma_init": getattr(args, "frozen_region_gamma_init", 0.003),
        "frozen_region_gamma_max": getattr(args, "frozen_region_gamma_max", 0.02),
        "frozen_region_uncertainty_threshold": getattr(args, "frozen_region_uncertainty_threshold", 0.35),
        "frozen_region_gate_temperature": getattr(args, "frozen_region_gate_temperature", 0.1),
        "frozen_region_trust_init": getattr(args, "frozen_region_trust_init", 0.05),
        "use_frozen_counterfactual_router": getattr(
            args, "use_frozen_counterfactual_router", False
        ),
        "frozen_counterfactual_aux_only": getattr(
            args, "frozen_counterfactual_aux_only", False
        ),
        "frozen_counterfactual_hidden_dim": getattr(
            args, "frozen_counterfactual_hidden_dim", 32
        ),
        "frozen_counterfactual_class_embed_dim": getattr(
            args, "frozen_counterfactual_class_embed_dim", 8
        ),
        "frozen_counterfactual_rho_init": getattr(
            args, "frozen_counterfactual_rho_init", 0.05
        ),
        "frozen_counterfactual_rho_max": getattr(
            args, "frozen_counterfactual_rho_max", 0.2
        ),
        "frozen_counterfactual_rescue_init": getattr(
            args, "frozen_counterfactual_rescue_init", 0.1
        ),
        "frozen_counterfactual_delta_clip": getattr(
            args, "frozen_counterfactual_delta_clip", 6.0
        ),
        "frozen_counterfactual_region_level": getattr(
            args, "frozen_counterfactual_region_level", ""
        ),
        "frozen_counterfactual_region_projection_dim": getattr(
            args, "frozen_counterfactual_region_projection_dim", 32
        ),
        "frozen_counterfactual_region_temperature": getattr(
            args, "frozen_counterfactual_region_temperature", 0.2
        ),
        "use_frozen_region_interaction_moe": getattr(
            args, "use_frozen_region_interaction_moe", False
        ),
        "frozen_region_interaction_aux_only": getattr(
            args, "frozen_region_interaction_aux_only", False
        ),
        "frozen_region_interaction_levels": tuple(getattr(
            args, "frozen_region_interaction_levels", ("C3", "C4")
        )),
        "frozen_region_interaction_projection_dim": getattr(
            args, "frozen_region_interaction_projection_dim", 64
        ),
        "frozen_region_interaction_temperature": getattr(
            args, "frozen_region_interaction_temperature", 0.2
        ),
        "frozen_region_interaction_shared_axis": getattr(
            args, "frozen_region_interaction_shared_axis", "width"
        ),
        "frozen_region_interaction_axis_radius": getattr(
            args, "frozen_region_interaction_axis_radius", 1
        ),
        "frozen_region_interaction_residual_init": getattr(
            args, "frozen_region_interaction_residual_init", 0.03
        ),
        "frozen_region_interaction_residual_max": getattr(
            args, "frozen_region_interaction_residual_max", 0.2
        ),
        "frozen_region_interaction_router_hidden_dim": getattr(
            args, "frozen_region_interaction_router_hidden_dim", 128
        ),
        "frozen_region_interaction_class_embed_dim": getattr(
            args, "frozen_region_interaction_class_embed_dim", 16
        ),
        "frozen_region_interaction_rho_init": getattr(
            args, "frozen_region_interaction_rho_init", 0.08
        ),
        "frozen_region_interaction_rho_max": getattr(
            args, "frozen_region_interaction_rho_max", 0.25
        ),
        "frozen_region_interaction_rescue_init": getattr(
            args, "frozen_region_interaction_rescue_init", 0.05
        ),
        "frozen_region_interaction_delta_clip": getattr(
            args, "frozen_region_interaction_delta_clip", 6.0
        ),
        "frozen_region_interaction_include_counterfactual": getattr(
            args,
            "frozen_region_interaction_include_counterfactual",
            False,
        ),
        "use_dvcre": getattr(args, "use_dvcre", False),
        "dvcre_levels": tuple(getattr(args, "dvcre_levels", ("C3", "C4"))),
        "dvcre_projection_dim": getattr(args, "dvcre_projection_dim", 64),
        "dvcre_topk_ratio": getattr(args, "dvcre_topk_ratio", 0.25),
        "dvcre_temperature": getattr(args, "dvcre_temperature", 0.2),
        "dvcre_residual_init": getattr(args, "dvcre_residual_init", 0.05),
        "dvcre_residual_max": getattr(args, "dvcre_residual_max", 0.2),
        "use_iscvf": getattr(args, "use_iscvf", False),
        "iscvf_levels": tuple(
            getattr(args, "iscvf_levels", ("C3", "C4", "C5"))
        ),
        "iscvf_gate_reduction": getattr(args, "iscvf_gate_reduction", 16),
        "iscvf_keep_prob": getattr(args, "iscvf_keep_prob", 0.75),
        "use_visual_evidence_router": getattr(
            args, "use_visual_evidence_router", False
        ),
        "visual_route_mode": getattr(args, "visual_route_mode", "pg_cver"),
        "visual_route_level": getattr(args, "visual_route_level", "C4"),
        "visual_route_projection_dim": getattr(
            args, "visual_route_projection_dim", 64
        ),
        "visual_route_topk_ratio": getattr(args, "visual_route_topk_ratio", 0.25),
        "visual_route_temperature": getattr(args, "visual_route_temperature", 0.2),
        "visual_route_shared_axis": getattr(
            args, "visual_route_shared_axis", "width"
        ),
        "visual_route_axis_radius": getattr(args, "visual_route_axis_radius", 1),
        "visual_route_gate_init": getattr(args, "visual_route_gate_init", 0.05),
        "visual_route_gamma_init": getattr(args, "visual_route_gamma_init", 0.05),
        "visual_route_gamma_max": getattr(args, "visual_route_gamma_max", 0.25),
        "visual_route_reject_temperature": getattr(
            args, "visual_route_reject_temperature", 0.1
        ),
        "visual_route_dropout": getattr(args, "visual_route_dropout", 0.1),
        "use_p9_caprs": getattr(args, "use_p9_caprs", False),
        "use_p10_wgcr": getattr(args, "use_p10_wgcr", False),
        "use_p11_otcvr": getattr(args, "use_p11_otcvr", False),
        "use_p12_berf": getattr(args, "use_p12_berf", False),
        "use_p13_vdrm": getattr(args, "use_p13_vdrm", False),
        "use_p14_hcaer": getattr(args, "use_p14_hcaer", False),
        "use_p15_vtr": getattr(args, "use_p15_vtr", False),
        "use_p16_facgr": getattr(args, "use_p16_facgr", False),
        "use_p17_dcasr": getattr(args, "use_p17_dcasr", False),
        "use_p18_ewsar": getattr(args, "use_p18_ewsar", False),
        "use_p19_apcer": getattr(args, "use_p19_apcer", False),
        "use_p20_cvcr": getattr(args, "use_p20_cvcr", False),
        "plain_innovation_levels": tuple(
            getattr(args, "plain_innovation_levels", ("C4", "C5"))
        ),
        "plain_innovation_projection_dim": getattr(
            args, "plain_innovation_projection_dim", 64
        ),
        "plain_innovation_topk": getattr(args, "plain_innovation_topk", 8),
        "plain_innovation_region_pooling": getattr(args, "plain_innovation_region_pooling", "topk"),
        "plain_innovation_temperature": getattr(
            args, "plain_innovation_temperature", 0.2
        ),
        "plain_innovation_dropout": getattr(
            args, "plain_innovation_dropout", 0.1
        ),
        "plain_innovation_gamma_init": getattr(
            args, "plain_innovation_gamma_init", 0.005
        ),
        "plain_innovation_gamma_max": getattr(
            args, "plain_innovation_gamma_max", 0.05
        ),
        "plain_innovation_base_floor": getattr(
            args, "plain_innovation_base_floor", 0.8
        ),
        "plain_innovation_use_counterfactual_experts": getattr(
            args, "plain_innovation_use_counterfactual_experts", True
        ),
        "plain_innovation_use_learned_router": getattr(
            args, "plain_innovation_use_learned_router", True
        ),
        "plain_innovation_router_variant": getattr(
            args, "plain_innovation_router_variant", "legacy"
        ),
        "plain_innovation_gamma_trainable": getattr(
            args, "plain_innovation_gamma_trainable", True
        ),
        "plain_innovation_warmup_epochs": getattr(
            args, "plain_innovation_warmup_epochs", 15
        ),
        "plain_innovation_ramp_epochs": getattr(
            args, "plain_innovation_ramp_epochs", 10
        ),
        "plain_innovation_sinkhorn_iters": getattr(
            args, "plain_innovation_sinkhorn_iters", 4
        ),
        "plain_innovation_router_start_epoch": getattr(
            args, "plain_innovation_router_start_epoch", 8
        ),
        "plain_innovation_router_ramp_epochs": getattr(
            args, "plain_innovation_router_ramp_epochs", 3
        ),
        "plain_innovation_gate_init": getattr(
            args, "plain_innovation_gate_init", 0.15
        ),
        "plain_innovation_trust_threshold": getattr(
            args, "plain_innovation_trust_threshold", 0.5
        ),
        "plain_innovation_trust_temperature": getattr(
            args, "plain_innovation_trust_temperature", 0.1
        ),
        "plain_innovation_uncertainty_floor": getattr(
            args, "plain_innovation_uncertainty_floor", 0.25
        ),
        "plain_innovation_budget_target": getattr(
            args, "plain_innovation_budget_target", 0.12
        ),
        "plain_innovation_advantage_temperature": getattr(
            args, "plain_innovation_advantage_temperature", 0.05
        ),
        "plain_innovation_gain_margin": getattr(
            args, "plain_innovation_gain_margin", 1e-5
        ),
        "plain_innovation_score_temperature": getattr(
            args, "plain_innovation_score_temperature", 1.0
        ),
        "plain_innovation_corrupt_probability": getattr(
            args, "plain_innovation_corrupt_probability", 0.75
        ),
        "plain_innovation_corrupt_ratio": getattr(
            args, "plain_innovation_corrupt_ratio", 0.25
        ),
        "plain_innovation_full_view_drop_probability": getattr(
            args, "plain_innovation_full_view_drop_probability", 0.35
        ),
        "plain_innovation_complement_margin": getattr(
            args, "plain_innovation_complement_margin", 0.02
        ),
        "plain_innovation_complement_temperature": getattr(
            args, "plain_innovation_complement_temperature", 0.05
        ),
    }

    filtered = {k: v for k, v in candidate_kwargs.items() if k in valid_keys}
    model = ConvNeXtV2Dual(**filtered)
    return model


def _safe_evaluate(data_loader_val, model_to_eval, device, amp=True, class_names=None, threshold=None, csv_path=None):
    try:
        return evaluate(
            data_loader_val, model_to_eval, device, amp=amp,
            class_names=class_names, threshold=threshold, csv_path=csv_path
        )
    except TypeError:
        try:
            return evaluate(
                data_loader_val, model_to_eval, device, amp=amp,
                class_names=class_names, threshold=threshold
            )
        except TypeError:
            try:
                return evaluate(data_loader_val, model_to_eval, device, amp=amp)
            except TypeError:
                return evaluate(data_loader_val, model_to_eval, device)


def _load_resume_checkpoint(model, optimizer, model_ema, scheduler, checkpoint_path, args):
    """加载续训检查点"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0, -1.0, 0

    print(f"📂 加载续训检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if args.resume_optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("✅ 优化器状态已恢复")

    if args.resume_scheduler and scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("✅ 学习率调度器状态已恢复")

    if model_ema is not None and 'model_ema' in checkpoint:
        model_ema.ema_state = checkpoint['model_ema']
        print("✅ EMA模型状态已恢复")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('metric', {}).get('mAP', -1.0)
    epochs_since_best = checkpoint.get('epochs_since_best', 0)

    print(f"✅ 从 {checkpoint_path} 成功续训。")
    print(f"   - 起始轮次: {start_epoch}")
    print(f"   - 已达最佳 mAP: {best_metric:.4f}")
    print(f"   - 早停计数器状态: {epochs_since_best}")

    return start_epoch, best_metric, epochs_since_best


def main(args):
    print(args)
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
    device = torch.device(args.device)

    try:
        mp.set_sharing_strategy('file_system')
        print("✅ 共享内存策略设置为 'file_system'")
    except Exception as e:
        print(f"⚠️ 设置共享内存策略时出错: {e}")

    torch.backends.cudnn.benchmark = not getattr(args, "deterministic", False)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("✅ CUDA 内存优化已启用")

    if getattr(args, "sem_conflict_lambda", 0.0) > 0 and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ sem_conflict_loss 已启用，自动设置 return_intermediate=True 以读取 logits_base/logits_sem")
    if getattr(args, "sem_trust_lambda", 0.0) > 0 and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ semantic trust supervision 已启用，自动设置 return_intermediate=True")
    if getattr(args, "sem_rank_lambda", 0.0) > 0 and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ semantic rank calibration 已启用，自动设置 return_intermediate=True")
    if getattr(args, "sem_error_lambda", 0.0) > 0 and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ visual-error semantic supervision 已启用，自动设置 return_intermediate=True")
    if getattr(args, "spatial_query_lambda", 0.0) > 0 and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ spatial-query supervision 已启用，自动设置 return_intermediate=True")
    view_evidence_enabled = (
        getattr(args, "view_evidence_aux_weight", 0.0) > 0
        or getattr(args, "view_evidence_distill_weight", 0.0) > 0
    )
    if view_evidence_enabled and not getattr(args, "use_view_evidence_distill", False):
        raise ValueError(
            "view-evidence loss weights require --use_view_evidence_distill true"
        )
    if view_evidence_enabled and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ cross-view best-evidence distillation 已启用，自动设置 return_intermediate=True")
    selective_rescue_enabled = any(float(getattr(args, name, 0.0)) > 0 for name in (
        "selective_rescue_aux_weight", "selective_rescue_loss_weight",
        "selective_rescue_guard_weight"))
    if selective_rescue_enabled and not getattr(args, "use_selective_view_rescue", False):
        raise ValueError("selective-rescue loss weights require --use_selective_view_rescue true")
    if getattr(args, "use_selective_view_rescue", False):
        args.return_intermediate = True
    frozen_rescue_enabled = any(float(getattr(args, name, 0.0)) > 0 for name in (
        "frozen_rescue_aux_weight", "frozen_rescue_trust_weight",
        "frozen_rescue_rank_weight", "frozen_rescue_guard_weight"))
    if frozen_rescue_enabled and not getattr(args, "use_frozen_anchor_rescue", False):
        raise ValueError("frozen-rescue losses require --use_frozen_anchor_rescue true")
    if getattr(args, "use_frozen_anchor_rescue", False):
        args.return_intermediate = True
    frozen_region_enabled = any(float(getattr(args, name, 0.0)) > 0 for name in (
        "frozen_region_aux_weight", "frozen_region_trust_weight",
        "frozen_region_rank_weight", "frozen_region_guard_weight"))
    if frozen_region_enabled and not getattr(args, "use_frozen_region_rescue", False):
        raise ValueError("frozen-region losses require --use_frozen_region_rescue true")
    if getattr(args, "use_frozen_region_rescue", False):
        args.return_intermediate = True
    frozen_counterfactual_enabled = any(
        float(getattr(args, name, 0.0)) > 0
        for name in (
            "frozen_counterfactual_route_weight",
            "frozen_counterfactual_rank_weight",
            "frozen_counterfactual_guard_weight",
            "frozen_counterfactual_residual_weight",
        )
    )
    if frozen_counterfactual_enabled and not getattr(
        args, "use_frozen_counterfactual_router", False
    ):
        raise ValueError(
            "counterfactual loss weights require "
            "--use_frozen_counterfactual_router true"
        )
    if getattr(args, "use_frozen_counterfactual_router", False):
        args.return_intermediate = True
    frozen_region_interaction_enabled = any(
        float(getattr(args, name, 0.0)) > 0
        for name in (
            "frozen_region_interaction_expert_weight",
            "frozen_region_interaction_alignment_weight",
            "frozen_region_interaction_diversity_weight",
            "frozen_region_interaction_route_weight",
            "frozen_region_interaction_rank_weight",
            "frozen_region_interaction_guard_weight",
            "frozen_region_interaction_residual_weight",
        )
    )
    if frozen_region_interaction_enabled and not getattr(
        args, "use_frozen_region_interaction_moe", False
    ):
        raise ValueError("M9 losses require --use_frozen_region_interaction_moe true")
    if getattr(args, "use_frozen_region_interaction_moe", False):
        args.return_intermediate = True
    if getattr(args, "frozen_region_interaction_router_only", False):
        if not getattr(args, "use_frozen_region_interaction_moe", False):
            raise ValueError("M9 router-only mode requires the M9 branch")
        if not getattr(args, "frozen_region_interaction_freeze_base", False):
            raise ValueError("M9 router-only mode requires a frozen R2 base")
        if getattr(args, "frozen_region_interaction_aux_only", False):
            raise ValueError("M9 router-only mode cannot be auxiliary-only")
    if getattr(args, "dvcre_aux_weight", 0.0) > 0 and not getattr(
        args, "use_dvcre", False
    ):
        raise ValueError("dvcre_aux_weight requires --use_dvcre true")
    iscvf_loss_enabled = (
        getattr(args, "iscvf_intervention_weight", 0.0) > 0
        or getattr(args, "iscvf_consistency_weight", 0.0) > 0
    )
    if iscvf_loss_enabled and not getattr(args, "use_iscvf", False):
        raise ValueError("IS-CVF loss weights require --use_iscvf true")
    if (
        getattr(args, "use_dvcre", False)
        or getattr(args, "use_iscvf", False)
    ) and not getattr(args, "return_intermediate", False):
        args.return_intermediate = True
        print("✅ DV-CRE/IS-CVF 已启用，自动设置 return_intermediate=True")
    visual_route_loss_enabled = any(
        float(getattr(args, name, 0.0)) > 0
        for name in (
            "visual_route_aux_weight",
            "visual_route_rescue_weight",
            "visual_route_guard_weight",
            "visual_route_cycle_weight",
        )
    )
    if visual_route_loss_enabled and not getattr(
        args, "use_visual_evidence_router", False
    ):
        raise ValueError(
            "visual-route loss weights require --use_visual_evidence_router true"
        )
    if (
        float(getattr(args, "visual_route_cycle_weight", 0.0)) > 0
        and getattr(args, "visual_route_mode", "pg_cver") != "cycle_cver"
    ):
        raise ValueError(
            "visual_route_cycle_weight requires --visual_route_mode cycle_cver"
        )
    if getattr(args, "use_visual_evidence_router", False) and not getattr(
        args, "return_intermediate", False
    ):
        args.return_intermediate = True
        print("✅ visual evidence routing 已启用，自动设置 return_intermediate=True")

    innovation_flag_names = (
        "use_p9_caprs",
        "use_p10_wgcr",
        "use_p11_otcvr",
        "use_p12_berf",
        "use_p13_vdrm",
        "use_p14_hcaer",
        "use_p15_vtr",
        "use_p16_facgr",
        "use_p17_dcasr",
        "use_p18_ewsar",
        "use_p19_apcer",
        "use_p20_cvcr",
    )
    enabled_innovations = [
        name for name in innovation_flag_names if getattr(args, name, False)
    ]
    if len(enabled_innovations) > 1:
        raise ValueError(
            "enable only one P9-P20 branch per experiment: "
            + ", ".join(enabled_innovations)
        )
    innovation_loss_names = (
        "plain_innovation_aux_weight",
        "plain_innovation_route_weight",
        "plain_innovation_guard_weight",
        "plain_innovation_consistency_weight",
        "plain_innovation_regret_weight",
        "plain_innovation_evidence_weight",
        "plain_innovation_match_weight",
        "plain_innovation_single_weight",
        "plain_innovation_rank_weight",
        "plain_innovation_budget_weight",
    )
    if any(
        float(getattr(args, name, 0.0)) > 0.0
        for name in innovation_loss_names
    ) and not enabled_innovations:
        raise ValueError("P9-P20 loss weights require one enabled innovation")
    if enabled_innovations:
        if getattr(args, "fuse_mode", "add") != "add":
            raise ValueError("P9-P20 branches require --fuse_mode add")
        if int(getattr(args, "plain_innovation_topk", 8)) <= 0:
            raise ValueError("plain_innovation_topk must be positive")
        if float(getattr(
            args, "plain_innovation_advantage_temperature", 0.05
        )) <= 0.0:
            raise ValueError(
                "plain_innovation_advantage_temperature must be positive"
            )
        if float(getattr(
            args, "plain_innovation_gain_margin", 1e-5
        )) < 0.0:
            raise ValueError("plain_innovation_gain_margin must be non-negative")
        if float(getattr(
            args, "plain_innovation_score_temperature", 1.0
        )) <= 0.0:
            raise ValueError(
                "plain_innovation_score_temperature must be positive"
            )
        if not 0.0 <= float(
            getattr(args, "plain_innovation_base_floor", 0.8)
        ) < 1.0:
            raise ValueError("plain_innovation_base_floor must be in [0, 1)")
        for name in (
            "plain_innovation_corrupt_probability",
            "plain_innovation_corrupt_ratio",
            "plain_innovation_full_view_drop_probability",
        ):
            if not 0.0 <= float(getattr(args, name, 0.0)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(getattr(
            args, "plain_innovation_complement_margin", 0.02
        )) < 0.0:
            raise ValueError(
                "plain_innovation_complement_margin must be non-negative"
            )
        if float(getattr(
            args, "plain_innovation_complement_temperature", 0.05
        )) <= 0.0:
            raise ValueError(
                "plain_innovation_complement_temperature must be positive"
            )
        args.return_intermediate = True
        print(
            "✅ Plain-BCE innovation enabled: "
            f"{enabled_innovations[0].removeprefix('use_')}"
        )
    if getattr(args, "plain_innovation_freeze_base", False):
        p9_router_only = (
            enabled_innovations == ["use_p9_caprs"]
            and getattr(args, "plain_innovation_router_only", False)
        )
        if not p9_router_only and enabled_innovations not in (
            ["use_p16_facgr"], ["use_p17_dcasr"], ["use_p19_apcer"]
        ):
            raise ValueError(
                "plain_innovation_freeze_base requires P16, P17, P19, "
                "or P9 residual router-only mode"
            )
        if not getattr(args, "finetune", ""):
            raise ValueError(
                "frozen plain innovation requires --finetune with a checkpoint"
            )
    if getattr(args, "plain_innovation_router_only", False):
        if enabled_innovations != ["use_p9_caprs"]:
            raise ValueError("plain_innovation_router_only requires P9")
        if not getattr(args, "plain_innovation_freeze_base", False):
            raise ValueError(
                "plain_innovation_router_only requires freeze_base=true"
            )
        if getattr(args, "plain_innovation_router_variant", "legacy") not in (
            "residual_nonbase",
            "selective_regret",
        ):
            raise ValueError(
                "plain_innovation_router_only requires residual_nonbase or "
                "selective_regret"
            )
        if not getattr(args, "plain_innovation_use_learned_router", True):
            raise ValueError(
                "plain_innovation_router_only requires a learned router"
            )

    reset_p9_router = getattr(
        args, "plain_innovation_reset_router_after_finetune", False
    )
    transfer_legacy_router = getattr(
        args, "plain_innovation_transfer_legacy_router", False
    )
    if reset_p9_router and transfer_legacy_router:
        raise ValueError(
            "P9 router reset and legacy transfer are mutually exclusive"
        )
    if reset_p9_router or transfer_legacy_router:
        if enabled_innovations != ["use_p9_caprs"]:
            raise ValueError("P9 router reset/transfer requires P9")
        if not getattr(args, "finetune", ""):
            raise ValueError("P9 router reset/transfer requires --finetune")
        if not getattr(args, "plain_innovation_use_learned_router", True):
            raise ValueError("P9 router reset/transfer requires a learned router")
    if transfer_legacy_router and getattr(
        args, "plain_innovation_router_variant", "legacy"
    ) != "selective_regret":
        raise ValueError(
            "legacy router transfer requires selective_regret"
        )

    loaders = _try_build_loaders_with_project(args)
    if len(loaders) == 3:
        data_loader_train, data_loader_val, class_names = loaders
    else:
        data_loader_train, data_loader_val = loaders
        class_names = None

    print(" assembling student model...")
    model = build_model(args).to(device)
    finetune_state = None
    if args.finetune:
        print(f"Load pre-trained student weights from: {args.finetune}")
        finetune_state = _load_finetune_weights(
            model, args.finetune, prefix=args.model_prefix or ''
        )
    if transfer_legacy_router:
        if finetune_state is None:
            raise RuntimeError("legacy router transfer could not load checkpoint")
        if not hasattr(model, "transfer_plain_innovation_legacy_router"):
            raise RuntimeError("selected model cannot transfer a P9 router")
        model.transfer_plain_innovation_legacy_router(finetune_state)
        print("[P9Router] transferred legacy non-base rows into utilities")
    if reset_p9_router:
        if not hasattr(model, "reset_plain_innovation_router"):
            raise RuntimeError("selected model cannot reset a P9 router")
        model.reset_plain_innovation_router()
        print("[P9Router] reset after finetune load")
    if getattr(args, "plain_innovation_freeze_base", False):
        if not hasattr(model, "freeze_base_for_plain_innovation"):
            raise RuntimeError("The selected model does not support frozen P16/P17/P19")
        model.freeze_base_for_plain_innovation()
        if getattr(args, "plain_innovation_router_only", False):
            if not hasattr(
                model, "freeze_plain_innovation_experts_for_router"
            ):
                raise RuntimeError(
                    "The selected model does not support P9 router-only mode"
                )
            model.freeze_plain_innovation_experts_for_router()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[PlainInnovationFrozenAnchor] frozen params={frozen:,}; "
            f"trainable branch params={trainable:,}; "
            f"router_only={getattr(args, 'plain_innovation_router_only', False)}"
        )
    if getattr(args, "sem_freeze_base", False):
        if not hasattr(model, "freeze_base_for_semantic"):
            raise RuntimeError("The selected model does not support semantic base freezing")
        model.freeze_base_for_semantic()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[SemanticAnchor] frozen base params={frozen:,}; trainable semantic params={trainable:,}")
    if getattr(args, "spatial_query_freeze_base", False):
        if not hasattr(model, "freeze_base_for_spatial_query"):
            raise RuntimeError("The selected model does not support spatial-query base freezing")
        model.freeze_base_for_spatial_query()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[SpatialQueryAnchor] frozen base params={frozen:,}; "
            f"trainable query params={trainable:,}"
        )
    if getattr(args, "frozen_rescue_freeze_base", False):
        if not hasattr(model, "freeze_base_for_frozen_rescue"):
            raise RuntimeError("The selected model does not support frozen rescue")
        model.freeze_base_for_frozen_rescue()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[FrozenRescueAnchor] frozen base params={frozen:,}; "
            f"trainable rescue params={trainable:,}"
        )
    if getattr(args, "frozen_region_freeze_base", False):
        if not hasattr(model, "freeze_base_for_frozen_region_rescue"):
            raise RuntimeError("The selected model does not support frozen region rescue")
        model.freeze_base_for_frozen_region_rescue()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[FrozenRegionAnchor] frozen base params={frozen:,}; "
            f"trainable region params={trainable:,}"
        )
    if getattr(args, "frozen_counterfactual_freeze_base", False):
        if not hasattr(model, "freeze_base_for_frozen_counterfactual_router"):
            raise RuntimeError(
                "The selected model does not support frozen counterfactual routing"
            )
        model.freeze_base_for_frozen_counterfactual_router()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[FrozenCounterfactualAnchor] frozen base params={frozen:,}; "
            f"trainable router params={trainable:,}"
        )
    if getattr(args, "frozen_region_interaction_freeze_base", False):
        if not hasattr(model, "freeze_base_for_frozen_region_interaction_moe"):
            raise RuntimeError("The selected model does not support M9")
        model.freeze_base_for_frozen_region_interaction_moe()
        if getattr(args, "frozen_region_interaction_router_only", False):
            model.freeze_region_interaction_experts_for_router()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        stage = "router" if getattr(
            args, "frozen_region_interaction_router_only", False
        ) else "experts"
        print(
            f"[M9RegionInteraction:{stage}] frozen params={frozen:,}; "
            f"trainable params={trainable:,}"
        )

    base_criterion = build_base_criterion(args)
    print(f"[BaseLoss] Using {args.base_loss}  ->  {base_criterion.__class__.__name__}")
        # GSPF wrapper 必须始终启用：
    # 即使 lambda=0，也需要在每个 batch 后清空 GSPF cache，避免显存持续上涨。
    base_criterion = GSPFRegularizedCriterion(
        base_criterion=base_criterion,
        lambda_consistency=args.gspf_lambda_consistency,
        lambda_ortho=args.gspf_lambda_ortho,
        cv_lambda_sem=args.cv_lambda_sem,
        cv_lambda_geo=args.cv_lambda_geo,
    )
    print(
        f"[Regularization] wrapper enabled: "
        f"gspf_consistency={args.gspf_lambda_consistency}, "
        f"gspf_ortho={args.gspf_lambda_ortho}, "
        f"cv_sem={args.cv_lambda_sem}, "
        f"cv_geo={args.cv_lambda_geo}"
    )

    if args.use_distillation:
        print("🔥 Knowledge distillation mode enabled!")
        teacher_build_args = argparse.Namespace(**vars(args))
        teacher_build_args.model = args.teacher_model
        teacher_model = build_model(teacher_build_args).to(device)

        if args.teacher_weights:
            print(f"   - Loading teacher weights from: {args.teacher_weights}")
            _load_finetune_weights(teacher_model, args.teacher_weights)
        else:
            print("   - ⚠️ WARNING: No teacher weights provided. Teacher will use random weights.")

        from models.modules.losses import DistillationLoss
        criterion = DistillationLoss(
            base_criterion=base_criterion,
            student_model=model,
            teacher_model=teacher_model,
            kd_mode=args.kd_mode,
            alpha=args.distillation_alpha,
            beta=args.distillation_beta,
            dkd_alpha=args.dkd_alpha,
            dkd_beta=args.dkd_beta,
            tau=args.distillation_tau,
            feature_layers=args.distill_feature_layers,
            adapter_configs=None,
        )
        print(f"   - DistillationLoss ready (kd_mode={args.kd_mode}, alpha={args.distillation_alpha}, tau={args.distillation_tau})")
    else:
        print("🔷 Standard training mode.")
        criterion = base_criterion

    print("Criterion =", criterion.__class__.__name__)
    criterion.sem_conflict_lambda = float(getattr(args, "sem_conflict_lambda", 0.0))
    criterion.sem_conflict_base_margin = float(getattr(args, "sem_conflict_base_margin", 0.0))
    criterion.sem_conflict_sem_margin = float(getattr(args, "sem_conflict_sem_margin", 0.0))
    criterion.sem_trust_lambda = float(getattr(args, "sem_trust_lambda", 0.0))
    criterion.sem_trust_temperature = float(getattr(args, "sem_trust_temperature", 0.01))
    criterion.sem_trust_margin = float(getattr(args, "sem_trust_margin", 0.0))
    criterion.sem_trust_target_mode = str(getattr(args, "sem_trust_target_mode", "soft"))
    criterion.sem_trust_harm_weight = float(getattr(args, "sem_trust_harm_weight", 3.0))
    criterion.sem_rank_lambda = float(getattr(args, "sem_rank_lambda", 0.0))
    criterion.sem_rank_guard_weight = float(getattr(args, "sem_rank_guard_weight", 2.0))
    criterion.sem_rank_temperature = float(getattr(args, "sem_rank_temperature", 0.2))
    criterion.sem_rank_need_temperature = float(getattr(args, "sem_rank_need_temperature", 0.2))
    criterion.sem_error_lambda = float(getattr(args, "sem_error_lambda", 0.0))
    criterion.sem_error_power = float(getattr(args, "sem_error_power", 2.0))
    criterion.sem_error_min_weight = float(getattr(args, "sem_error_min_weight", 0.05))
    criterion.sem_error_guard_weight = float(getattr(args, "sem_error_guard_weight", 4.0))
    criterion.spatial_query_lambda = float(getattr(args, "spatial_query_lambda", 0.0))
    criterion.spatial_query_final_weight = float(
        getattr(args, "spatial_query_final_weight", 0.25)
    )
    criterion.spatial_query_guard_weight = float(
        getattr(args, "spatial_query_guard_weight", 1.0)
    )
    criterion.spatial_query_negative_weight = float(
        getattr(args, "spatial_query_negative_weight", 1.0)
    )
    criterion.view_evidence_aux_weight = float(
        getattr(args, "view_evidence_aux_weight", 0.0)
    )
    criterion.view_evidence_distill_weight = float(
        getattr(args, "view_evidence_distill_weight", 0.0)
    )
    criterion.view_evidence_warmup_epochs = int(
        getattr(args, "view_evidence_warmup_epochs", 5)
    )
    criterion.view_evidence_ramp_epochs = int(
        getattr(args, "view_evidence_ramp_epochs", 5)
    )
    criterion.view_evidence_advantage_temperature = float(
        getattr(args, "view_evidence_advantage_temperature", 0.1)
    )
    criterion.view_evidence_negative_weight = float(
        getattr(args, "view_evidence_negative_weight", 1.0)
    )
    for name, default, cast in (
        ("selective_rescue_aux_weight", 0.0, float),
        ("selective_rescue_loss_weight", 0.0, float),
        ("selective_rescue_guard_weight", 0.0, float),
        ("selective_rescue_warmup_epochs", 5, int),
        ("selective_rescue_ramp_epochs", 5, int),
        ("selective_rescue_aux_decay_start", 30, int),
        ("selective_rescue_aux_decay_end", 80, int),
    ):
        setattr(criterion, name, cast(getattr(args, name, default)))
    for name, default in (
        ("frozen_rescue_aux_weight", 0.0),
        ("frozen_rescue_trust_weight", 0.0),
        ("frozen_rescue_rank_weight", 0.0),
        ("frozen_rescue_guard_weight", 0.0),
        ("frozen_rescue_rank_temperature", 0.2),
    ):
        setattr(criterion, name, float(getattr(args, name, default)))
    for name, default in (
        ("frozen_region_aux_weight", 0.0),
        ("frozen_region_trust_weight", 0.0),
        ("frozen_region_rank_weight", 0.0),
        ("frozen_region_guard_weight", 0.0),
        ("frozen_region_rank_temperature", 0.1),
        ("frozen_region_target_gain", 0.005),
    ):
        setattr(criterion, name, float(getattr(args, name, default)))
    for name, default in (
        ("frozen_counterfactual_route_weight", 0.0),
        ("frozen_counterfactual_rank_weight", 0.0),
        ("frozen_counterfactual_guard_weight", 0.0),
        ("frozen_counterfactual_residual_weight", 0.0),
        ("frozen_counterfactual_router_temperature", 0.02),
        ("frozen_counterfactual_router_margin", 0.02),
        ("frozen_counterfactual_rank_temperature", 0.2),
        ("frozen_counterfactual_rank_margin", 0.5),
        ("frozen_counterfactual_hard_threshold", 2.0),
        ("frozen_counterfactual_guard_threshold", 4.0),
    ):
        setattr(criterion, name, float(getattr(args, name, default)))
    criterion.frozen_counterfactual_queue_size = int(
        getattr(args, "frozen_counterfactual_queue_size", 128)
    )
    for name, default in (
        ("frozen_region_interaction_expert_weight", 0.0),
        ("frozen_region_interaction_alignment_weight", 0.0),
        ("frozen_region_interaction_diversity_weight", 0.0),
        ("frozen_region_interaction_route_weight", 0.0),
        ("frozen_region_interaction_rank_weight", 0.0),
        ("frozen_region_interaction_guard_weight", 0.0),
        ("frozen_region_interaction_residual_weight", 0.0),
        ("frozen_region_interaction_expert_temperature", 0.1),
        ("frozen_region_interaction_router_temperature", 0.03),
        ("frozen_region_interaction_router_margin", 0.01),
        ("frozen_region_interaction_rank_temperature", 0.2),
        ("frozen_region_interaction_rank_margin", 0.5),
        ("frozen_region_interaction_hard_threshold", 2.0),
        ("frozen_region_interaction_guard_threshold", 4.0),
    ):
        setattr(criterion, name, float(getattr(args, name, default)))
    criterion.frozen_region_interaction_queue_size = int(getattr(
        args, "frozen_region_interaction_queue_size", 128
    ))
    criterion.dvcre_aux_weight = float(getattr(args, "dvcre_aux_weight", 0.0))
    criterion.iscvf_intervention_weight = float(
        getattr(args, "iscvf_intervention_weight", 0.0)
    )
    criterion.iscvf_consistency_weight = float(
        getattr(args, "iscvf_consistency_weight", 0.0)
    )
    criterion.iscvf_warmup_epochs = int(
        getattr(args, "iscvf_warmup_epochs", 5)
    )
    criterion.iscvf_ramp_epochs = int(getattr(args, "iscvf_ramp_epochs", 5))
    criterion.visual_route_aux_weight = float(
        getattr(args, "visual_route_aux_weight", 0.0)
    )
    criterion.visual_route_rescue_weight = float(
        getattr(args, "visual_route_rescue_weight", 0.0)
    )
    criterion.visual_route_guard_weight = float(
        getattr(args, "visual_route_guard_weight", 0.0)
    )
    criterion.visual_route_cycle_weight = float(
        getattr(args, "visual_route_cycle_weight", 0.0)
    )
    criterion.visual_route_negative_weight = float(
        getattr(args, "visual_route_negative_weight", 1.0)
    )
    for name in (
        "plain_innovation_aux_weight",
        "plain_innovation_route_weight",
        "plain_innovation_guard_weight",
        "plain_innovation_consistency_weight",
        "plain_innovation_regret_weight",
        "plain_innovation_evidence_weight",
        "plain_innovation_match_weight",
        "plain_innovation_single_weight",
        "plain_innovation_rank_weight",
        "plain_innovation_budget_weight",
    ):
        setattr(criterion, name, float(getattr(args, name, 0.0)))
    if criterion.sem_conflict_lambda > 0:
        print(
            "[SemanticConflict] enabled: "
            f"lambda={criterion.sem_conflict_lambda}, "
            f"base_margin={criterion.sem_conflict_base_margin}, "
            f"sem_margin={criterion.sem_conflict_sem_margin}"
        )
    if criterion.sem_trust_lambda > 0:
        print(
            "[SemanticTrust] enabled: "
            f"lambda={criterion.sem_trust_lambda}, "
            f"temperature={criterion.sem_trust_temperature}, "
            f"margin={criterion.sem_trust_margin}, "
            f"target={criterion.sem_trust_target_mode}, "
            f"harm_weight={criterion.sem_trust_harm_weight}"
        )
    if criterion.sem_rank_lambda > 0:
        print(
            "[SemanticRank] enabled: "
            f"lambda={criterion.sem_rank_lambda}, "
            f"guard_weight={criterion.sem_rank_guard_weight}, "
            f"temperature={criterion.sem_rank_temperature}, "
            f"need_temperature={criterion.sem_rank_need_temperature}"
        )
    if criterion.sem_error_lambda > 0:
        print(
            "[SemanticError] enabled: "
            f"lambda={criterion.sem_error_lambda}, "
            f"power={criterion.sem_error_power}, "
            f"min_weight={criterion.sem_error_min_weight}, "
            f"guard_weight={criterion.sem_error_guard_weight}"
        )
    if criterion.spatial_query_lambda > 0:
        print(
            "[SpatialQueryLoss] enabled: "
            f"lambda={criterion.spatial_query_lambda}, "
            f"final_weight={criterion.spatial_query_final_weight}, "
            f"guard_weight={criterion.spatial_query_guard_weight}, "
            f"negative_weight={criterion.spatial_query_negative_weight}"
        )
    if (
        criterion.view_evidence_aux_weight > 0
        or criterion.view_evidence_distill_weight > 0
    ):
        print(
            "[CrossViewBestEvidence] enabled: "
            f"aux_weight={criterion.view_evidence_aux_weight}, "
            f"distill_weight={criterion.view_evidence_distill_weight}, "
            f"warmup={criterion.view_evidence_warmup_epochs}, "
            f"ramp={criterion.view_evidence_ramp_epochs}, "
            f"adv_temperature={criterion.view_evidence_advantage_temperature}, "
            f"negative_weight={criterion.view_evidence_negative_weight}"
        )
    if criterion.dvcre_aux_weight > 0:
        print(
            "[DV-CRE] enabled: "
            f"levels={getattr(args, 'dvcre_levels', ['C3', 'C4'])}, "
            f"topk={getattr(args, 'dvcre_topk_ratio', 0.25)}, "
            f"aux_weight={criterion.dvcre_aux_weight}"
        )
    if (
        criterion.iscvf_intervention_weight > 0
        or criterion.iscvf_consistency_weight > 0
    ):
        print(
            "[IS-CVF] enabled: "
            f"levels={getattr(args, 'iscvf_levels', ['C3', 'C4', 'C5'])}, "
            f"intervention_weight={criterion.iscvf_intervention_weight}, "
            f"consistency_weight={criterion.iscvf_consistency_weight}, "
            f"warmup={criterion.iscvf_warmup_epochs}, "
            f"ramp={criterion.iscvf_ramp_epochs}"
        )
    if getattr(args, "use_visual_evidence_router", False):
        print(
            "[VisualEvidenceRouter] enabled: "
            f"mode={getattr(args, 'visual_route_mode', 'pg_cver')}, "
            f"level={getattr(args, 'visual_route_level', 'C4')}, "
            f"aux={criterion.visual_route_aux_weight}, "
            f"rescue={criterion.visual_route_rescue_weight}, "
            f"guard={criterion.visual_route_guard_weight}, "
            f"cycle={criterion.visual_route_cycle_weight}"
        )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying freeze settings")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr_ratio=0.0):
        def lr_lambda(current_epoch):
            if current_epoch < num_warmup_epochs:
                return float(current_epoch) / float(max(1, num_warmup_epochs))
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)
        return LambdaLR(optimizer, lr_lambda)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_epochs=args.warmup_epochs,
        num_training_epochs=args.epochs,
        min_lr_ratio=args.min_lr / args.lr
    )

    print(f"✅ 学习率调度器已创建:")
    print(f"   - Warmup 轮次: {args.warmup_epochs}")
    print(f"   - 总训练轮次: {args.epochs}")
    print(f"   - 初始学习率: {args.lr}")
    print(f"   - 最小学习率: {args.min_lr}")

    model_ema = None
    if args.teacher_mode:
        model_ema = SimpleEMA(model, decay=args.ema_decay, device=args.ema_device)
        print(f"[teacher_mode] EMA enabled with decay={args.ema_decay}, device={args.ema_device}")

    start_epoch = 0
    best_metric = -1.0
    epochs_since_best = 0

    if args.resume:
        checkpoint_path = args.resume
        if os.path.isdir(checkpoint_path):
            last_ckpt = Path(checkpoint_path) / "checkpoint_last.pth"
            best_ckpt = Path(checkpoint_path) / "checkpoint_best.pth"
            if last_ckpt.exists():
                checkpoint_path = str(last_ckpt)
                print(f"检测到目录，使用最新的检查点: {checkpoint_path}")
            elif best_ckpt.exists():
                checkpoint_path = str(best_ckpt)
                print(f"检测到目录，使用最佳的检查点: {checkpoint_path}")
            else:
                print(f"⚠️ 续训目录 {args.resume} 为空，将从头开始训练。")
                checkpoint_path = None

        if checkpoint_path and os.path.isfile(checkpoint_path):
            start_epoch, best_metric, epochs_since_best = _load_resume_checkpoint(
                model, optimizer, model_ema, scheduler, checkpoint_path, args
            )

        if args.resume_epoch >= 0:
            print(f"手动覆盖起始轮次为: {args.resume_epoch}")
            start_epoch = args.resume_epoch

    best_val_stats = {}
    metric_name = "mAP"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_log.csv"

    if start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_metric', 'best_metric',
                'epoch_time', 'avg_epoch_time', 'estimated_remaining_hours',
                'completion_time', 'grad_var'
            ])
        print("📝 创建新的训练日志文件")
    else:
        print(f"📝 续训模式，将追加到现有日志文件: {csv_path}")

    import datetime
    start_time = time.time()
    epoch_times = []

    print(f"\n🎯 开始训练，总轮次: {args.epochs}, 起始轮次: {start_epoch}")
    if args.patience > 0:
        print(f"⌛ 早停机制已启用，耐心值 (Patience) = {args.patience} 轮")
        if args.early_stop_start_epoch > 0:
            print(
                "⌛ 早停计数将从 Epoch "
                f"{args.early_stop_start_epoch} 开始"
            )
    print(f"📊 训练集 batches/epoch: {len(data_loader_train)}")

    if start_epoch == 0 and getattr(args, "reseed_before_training", False):
        set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
        print(f"🔒 模型构建后重新固定 RNG: seed={args.seed}")

    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=True,
            model_ema=model_ema,
            accum_iter=args.accum_iter,
        )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch < args.warmup_epochs:
            print(f"📈 Epoch {epoch} 学习率: {current_lr:.2e}")

        if model_ema is not None:
            eval_model = build_model(args).to(device)
            model_ema.copy_to(eval_model)
            print("📊 使用EMA模型进行评估")
        else:
            eval_model = model
            print("📊 使用原始模型进行评估")

        val_stats = _safe_evaluate(
            data_loader_val=data_loader_val,
            model_to_eval=eval_model,
            device=device,
            amp=True,
            class_names=class_names,
            threshold=args.eval_threshold,
            csv_path=None,
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch - 1
        estimated_remaining = avg_epoch_time * remaining_epochs
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining)

        print(f"⏰ Epoch {epoch} 耗时: {epoch_time:.1f}s, 平均: {avg_epoch_time:.1f}s, 剩余预估: {estimated_remaining/3600:.1f}h")
        print(f"  预计完成: {completion_time.strftime('%m-%d %H:%M')}")

        primary = None
        if isinstance(val_stats, dict):
            for k in ["mAP", "map", "AP", "ap", "acc1", "acc", "top1"]:
                if k in val_stats:
                    primary = float(val_stats[k])
                    metric_name = k
                    break
        if primary is None:
            primary = -float(val_stats.get("loss", train_stats.get("loss", 0.0))) if isinstance(val_stats, dict) else -float(train_stats.get("loss", 0.0))
            metric_name = "-loss"

        is_best = primary > best_metric

        if is_best:
            best_metric = primary
            best_val_stats = val_stats
            epochs_since_best = 0
            print(f"🎉 新的最佳性能! mAP = {best_metric:.4f}. 重置早停计数器。")
        else:
            if epoch >= max(int(args.early_stop_start_epoch), 0):
                epochs_since_best += 1
                print(f"📉 性能未提升，早停计数器: {epochs_since_best}/{args.patience}")
            else:
                epochs_since_best = 0
                print(
                    "⏸️ 分支预训练阶段，早停尚未开始计数 "
                    f"({epoch}/{args.early_stop_start_epoch})"
                )

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats.get('loss', 0.0),
                primary,
                best_metric,
                epoch_time,
                avg_epoch_time,
                estimated_remaining / 3600,
                completion_time.strftime('%m-%d %H:%M'),
                train_stats.get('grad_var', 0.0)
            ])

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metric": {metric_name: best_metric},
            "val_stats": val_stats,
            "epochs_since_best": epochs_since_best
        }

        if model_ema is not None:
            ckpt["model_ema"] = model_ema.ema_state

        torch.save(ckpt, str(output_dir / "checkpoint_last.pth"))
        if is_best:
            torch.save(ckpt, str(output_dir / "checkpoint_best.pth"))

        took = time.time() - start
        print(f"[epoch {epoch}] val {metric_name}={primary:.4f} (best={best_metric:.4f})   elapsed={took/60.0:.1f} min")

        if (
            args.patience > 0
            and epoch >= max(int(args.early_stop_start_epoch), 0)
            and epochs_since_best >= args.patience
        ):
            print(f"\n🛑 触发早停! 验证集指标已连续 {args.patience} 轮未提升。")
            print(f"   - 最佳性能出现在第 {epoch - epochs_since_best} 轮，{metric_name} = {best_metric:.4f}")
            break

    total_time = time.time() - start_time
    print(f"\n✅ 训练完成! 总耗时: {total_time/3600:.2f} 小时")

    print("📊 按mAP排序训练日志...")
    try:
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values('val_metric', ascending=False)
        df_sorted.to_csv(csv_path, index=False)
        print(f"✅ 训练日志已按mAP排序并保存至: {csv_path}")
    except Exception as e:
        print(f"⚠️ 排序CSV文件时出错: {e}")

    try:
        total_params = sum(p.numel() for p in model.parameters())
    except Exception:
        total_params = 0

    best_per_class_ap = best_val_stats.get('per_class_ap', [])

    append_summary_to_global_log(
        args=args,
        best_metric_value=best_metric,
        metric_name=metric_name,
        model_total_params=total_params,
        class_names=class_names,
        per_class_ap_list=best_per_class_ap
    )

    print("Finished.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
