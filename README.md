# Uniform Fusion — 论文官方代码

> Class-Conditioned Multi-Scale Regional Learning for Dual-View X-Ray Multi-Label Recognition
> （对应稿件 `UniformFusion_submission_revision_20260919_tablebold.pdf`）

本仓库是上述论文的官方实现与复现代码，包含**主方法 Uniform Fusion**、全部对照实验的锁定源码、
运行脚本与数据划分清单。所有代码取自论文最终归档的冻结快照，SHA256 与论文各实验 `protocol.txt` 记录值一致。

| | |
|---|---|
| **任务** | 双视角（OL / SD）X 光违禁品多标签识别 |
| **主方法** | `resnet50` 双分支 + C4/C5 多尺度 class-query Top-K 区域证据 + 反事实专家 + 均匀聚合（内部配置名 `Final_NoAnchor_NoRouter`） |
| **数据集** | DvXray（15 类，12,800 / 1,600 / 1,600）、LDXray（12 类，99,133 / 11,015 / 36,849） |
| **环境** | Python 3.10.20 / PyTorch 2.9.0+cu128 / timm 1.0.22 |
| **主入口** | `bash main.sh`（一键：准备数据链接 → 自检 → 训练 → 锁定 Test 评估） |

---

## 快速开始（5 步）

```bash
# 0) 环境（详见第 3 节）
conda create -n v2b384_env python=3.10.20 -y && conda activate v2b384_env
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1) 准备标注（把 data_splits/ 展开成脚本期望的 core/annotations/ 布局）
bash main.sh setup

# 2) 把划分文件里的原始机器路径指向你的数据根（详见 §7.1）
RELINK_FROM=/home/hfuu/桌面/convnextv2/data RELINK_TO=/你的/DvXray_root bash main.sh relink

# 3) 自检（解释器/依赖/CUDA、锁定源码与划分 SHA256、数据集可访问性）
bash main.sh preflight

# 4) 训练主方法（训练集训练，Val 选最佳 checkpoint）
GPU_ID=0 SEED=930163947 bash main.sh train

# 5) 锁定 Test 评估一次（每个配置只允许一次，脚本会强制约束）
bash main.sh eval
```

先看将要执行什么而不真正运行：`DRY_RUN=true bash main.sh`。
完整子命令见 §7.2，逐表复现命令见 §6 与 §7.6，产物结构见 §7.8。

**范围**：本包只收录该 PDF 中实际出现的实验所对应的代码。论文未提及的方法（语义/LLM 分支、
判别性中心化、SIXray/CUB 跨数据集、p16–p20 候选头消融、Learned Router 鲁棒性、最优路由筛选等）
一律未收录。

**代码来源**：全部取自论文最终归档 `论文最终归档_20260830/` 的冻结快照。

---

## 1. 目录结构

```
UniformFusion_code_submission/
├── README.md                     本文件（含完整复现步骤）
├── main.sh                       ★ 一键入口：setup / relink / preflight / train / eval / status
├── LICENSE                       MIT License（`third_party/` 快照除外，见 §13）
├── CITATION.cff                  引用元数据（GitHub「Cite this repository」按钮由此生成）
├── environment_versions.txt      论文运行环境记录（conda 环境 v2b384_env）
├── requirements.txt              核心依赖清单（完整 81 包见 pip_freeze.txt）
├── pip_freeze.txt                环境完整 pip freeze（81 个包）
├── SHA256SUMS.txt                冻结清单（对应论文归档的 184 个文件）
├── SHA256SUMS.restored.txt       公开发布时从归档恢复的 24 个辅助工具哈希
├── .gitignore                    忽略运行产物 / 权重 / 缓存
│
├── core/                         ★ 论文锁定源码（可直接运行）
│   ├── main_finetune.py          训练主入口（354 个参数）
│   ├── engine_finetune.py        训练/评估引擎
│   ├── datasets.py               双视角数据集与条件增强策略
│   ├── utils.py, optim_factory.py
│   ├── models/
│   │   ├── convnextv2_dual.py                ★ 主模型
│   │   ├── modules/plain_bce_innovations.py  ★ 区域证据/专家/Guard/均匀聚合
│   │   ├── modules/visual_evidence.py        C4/C5 多尺度 class-query Top-K 选择
│   │   ├── modules/{fusions,necks,attentions,cross_view_consistency,augmentations,common,losses,composite_loss}.py
│   │   ├── modules/plain_bce_p16..p20.py     被 convnextv2_dual.py 顶层导入，必须保留
│   │   ├── modules/custom_losses/, modules/distillation/
│   │   ├── timm_backbones.py, tv_backbones.py, convnextv1.py, convnextv2.py, utils.py
│   │   └── official_ahcr_adapter.py, fair_baseline_adapters.py
│   └── tools/                    全部工具脚本（39 个）：评估、协议校验、统计、测速、
│                                 smoke 自检、结果汇总、图表构建（含从归档恢复的 24 个，见 §11）
│
├── data_splits/                  数据集划分列表（纯文本路径，不含任何图像）
│   ├── DvXray/{classes.txt, DvXray_train.txt, DvXray_val.txt, DvXray_test.txt}
│   └── LDXray/{ldxray_classes.txt, LDXray_train.txt, LDXray_val.txt, LDXray_test.txt, LDXray_split_manifest.json}
│
└── experiments/                  按论文条目分组的运行脚本 + 实验专用快照
    ├── 00_shared_runners/        被多组共用的单次运行器
    ├── 01_DvXray_main_n5/                 Table 1
    ├── 02_LDXray_cross_dataset/           Table 8
    ├── 03_ConvNeXtV2_backbone/            §4.2.3
    ├── 04_MaxViT_backbone/                §4.2.3
    ├── 05_component_analysis/             Table 4
    ├── 06_AHCR_comparison/                §4.2.2
    ├── 07_fusion_baselines/               Table 2
    ├── 08_adapter_baselines/              Table 3
    ├── 09_controls/                       Table 5
    ├── 10_correction_and_sensitivity/     Table 6 + §4.3.4 敏感性
    ├── 11_single_view/                    Table 7
    ├── 12_view_ablation/                  §4.3.4 视角错配
    ├── 13_internal_vs_final_analysis/     §4.3.3
    ├── 14_efficiency/                     Table 9
    └── 15_figures_and_predictions/        Figure 3 / Figure 4
```

---

## 2. 未包含的内容

| 缺失项 | 说明 |
|---|---|
| 数据集原图 | DvXray 与 LDXray 的 X 光图像 |
| 数据集标注 | LDXray 原始 JSON；DvXray 原始标注 |
| 预训练权重 | 如 ConvNeXtV2-Tiny 的 `student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors` |
| 训练产物 | checkpoint、`*_metrics.json/csv`、`train.log`、汇总 CSV |
| 论文图表 | PDF/PNG/SVG/热图，以及全部结果表 |
| 划分文件里的图像路径 | 列表保存的是原始机器的绝对路径（`/home/hfuu/...`）。代码不做路径重写，请按 §7.1 用 `main.sh relink` 或自行替换为你的数据根 |
| 辅助工具脚本 | 论文归档中另有 24 个 smoke 自检 / 结果汇总 / 图表构建工具未在原冻结包内，已随本次公开发布从归档恢复至 `core/tools/`（见 §11） |

> `data_splits/` 只提供**划分清单**（每行一个图像路径），用于确认划分与论文一致，不含图像本身。

---

## 3. 运行环境

### 3.1 从零创建环境（推荐）

```bash
conda create -n v2b384_env python=3.10.20 -y
conda activate v2b384_env

# PyTorch 官方轮子需指定 cu128 索引（论文用 2.9.0+cu128）
pip install torch==2.9.0 torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 其余核心依赖
pip install -r requirements.txt
```

### 3.2 复用已有环境

```bash
conda activate v2b384_env        # Python 3.10.20 / torch 2.9.0+cu128 / timm 1.0.22
pip install -r requirements.txt  # 完整环境见 pip_freeze.txt（81 个包）
```

### 3.3 论文测量环境（4.6 节）

Ubuntu 24.04.4 LTS、Python 3.10.20、PyTorch 2.9.0+cu128、CUDA Runtime 12.8、
cuDNN 9.10.2、NVIDIA GeForce RTX 5080 (16 GB)。机器可读版本见 `environment_versions.txt`。

- 单卡即可复现全部论文实验；`GPU_ID=<n>` 选择显卡。
- 单次训练显存占用约 6–8 GB（`--batch_size 32`、`resnet50`、`224²`、双视角）。
- 可选依赖（`try/except` 或延迟导入，不装也能跑主实验）：`tensorboardX`、`wandb`、`thop`、`apex`、`MinkowskiEngine`。
- 离线环境可设 `HF_HUB_OFFLINE=1`；`main.sh` 会自动导出 `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  与 `PYTORCH_ALLOC_CONF=expandable_segments:True`。

---

## 4. 核心代码版本锁定

以下三个文件是论文结果的唯一权威版本，其 SHA256 已写入论文 `protocol.txt`
（见 `experiments/01_DvXray_main_n5/protocol_uniformfusion_n5.txt`）。

| 文件 | SHA256 |
|---|---|
| `core/main_finetune.py` | `d1ec51b1e44152db52def599d2185d02318786de2db1177bff3158b26ce1782d` |
| `core/models/convnextv2_dual.py` | `f39a507d877dd8350fe862fdfc013032f7c872e6ca869371f9bbdb64fa81abcb` |
| `core/engine_finetune.py` | `dbe37f44aee890644a3016012b1b240a18ddd933e09fadb18d0ea38e7c1140ff` |

数据划分校验（与脚本内校验常量一致）：

| 文件 | SHA256 |
|---|---|
| `data_splits/DvXray/DvXray_train.txt` | `f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf` |
| `data_splits/DvXray/DvXray_val.txt` | `a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67` |
| `data_splits/DvXray/DvXray_test.txt` | `6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6` |

划分规模与论文 4.1.1 节一致：DvXray 12,800 / 1,600 / 1,600（15 类）；
LDXray 99,133 / 11,015 / 36,849（12 类，val 由 seed 20260901 从官方训练集抽样 10% 得到）。

---

## 5. 方法命名对照（务必先读）

| 论文中的名字 | 内部配置名 | 关键旗标 |
|---|---|---|
| **Uniform Fusion**（主方法） | `Final_NoAnchor_NoRouter` | `--plain_innovation_use_learned_router false` 且 `--plain_innovation_route_weight 0.0` |
| Learned Router（历史变体，论文未使用） | `Final_NoAnchor` | `use_learned_router true`，`route_weight 0.05` |

依据：`04_原始结果表/06_UniformFusion主方法_n5/README.md` 原文——
“`Final_NoAnchor` 在这些表中表示 Learned Router 历史完整变体；`Uniform_Fusion` 才是当前主方法。”

代码依据：`core/models/modules/plain_bce_innovations.py` 中当 `use_learned_router=False` 时，
专家权重直接取 `paired.new_full(..., 1.0 / expert_count)`，即均匀权重——这正是论文
3.4 节 “Uniform logit aggregation” 的实现。

> ⚠️ `run_final_noanchor_locked_pair_one.sh` 同时支持 `Plain_BCE` 与 `Final_NoAnchor`。
> 本包保留它是因为 **Plain_BCE 的 n=5 结果（Table 1 基线）由它产出**；
> 其中 `METHOD=Final_NoAnchor` 分支不属于论文，请勿使用。

---

## 6. 实验 → 论文章节对照

| 论文条目 | 目录 | 关键脚本 | 运行器 |
|---|---|---|---|
| Table 1 DvXray ResNet-50 n=5 | `01_DvXray_main_n5/` | `run_uniform_missing_2seeds.sh`（主方法补齐 2 seed）<br>`run_final_noanchor_resnet50_n5.sh`（Plain-BCE 基线） | `00_shared_runners/run_p9_final_noanchor_ablation_one.sh`<br>`01/run_final_noanchor_locked_pair_one.sh` |
| Table 2 融合配置（Mean/Max/Concat/Cross-attn） | `07_fusion_baselines/` | `run_fair_fusion_baselines_3seeds.sh` | 同目录 |
| Table 3 DAGNet / ML-Decoder 适配器 | `08_adapter_baselines/` | `run_fair_adapters_3seeds.sh`<br>`run_dagnet_batch32_recheck_3seeds.sh` | `08/run_fair_adapter_one.sh` |
| §4.2.2 AHCR 双协议 n=5 | `06_AHCR_comparison/` | `run_ahcr_official_strict_5seeds.sh`<br>`run_official_ahcr_resnet50_5seeds.sh` | `06/run_ahcr_official_strict_one.sh`、`tools/run_ahcr_official_strict.py` |
| §4.2.3 ConvNeXtV2-Tiny n=3 | `03_ConvNeXtV2_backbone/` | `run_convnextv2_uniform_fusion_3seeds.sh` | `03/run_convnextv2_uniform_fusion_one.sh` |
| §4.2.3 MaxViT-Tiny n=3 | `04_MaxViT_backbone/` | `run_maxvit_uniform_3seeds.sh` | `04/run_maxvit_uniform_one.sh` |
| Table 4 组件分析（去复制视图分支／去 guard／仅 C5） | `05_component_analysis/` | `run_uniform_component_ablation_3seeds.sh` | `05/run_uniform_component_ablation_one.sh` |
| Table 5 监督与空间选择对照（含 GAP／dense／NoAux） | `09_controls/` | `run_reviewer_first_batch_9runs.sh`<br>`run_uf_gap_3seeds.sh`、`run_uf_noaux_3seeds.sh` | `00_shared_runners/run_reviewer_first_batch_one.sh` |
| Table 6 / §4.3.4 修正路径对照与敏感性 | `10_correction_and_sensitivity/` | `run_uniform_residual_ablation_3seeds.sh`（NoCorrection/NoRamp）<br>`run_uniform_val_sensitivity.sh`（K/τ/γmax） | 同上 |
| Table 7 独立单视角基线 | `11_single_view/` | `run_single_view_plain_3seeds.sh` | 同目录 |
| §4.3.4 视角错配（OL 复制／SD 复制／循环错配） | `12_view_ablation/` | `run_final_noanchor_view_ablation_test_n5.sh` | 复用 `01/` 的 5 个锁定 checkpoint |
| §4.3.3 内部基分支 vs 最终残差（免训练） | `13_internal_vs_final_analysis/` | `analyze_uniform_internal_evidence.py` | — |
| Table 8 LDXray n=3 | `02_LDXray_cross_dataset/` | `run_ldxray_uniform_stepmatched_3seeds.sh` | `02/run_ldxray_uniform_one.sh` |
| Table 9 效率与跨 session 时延 | `14_efficiency/` | `run_final_noanchor_efficiency.sh`<br>`run_final_efficiency_5sessions.sh` | `14/tools/profile_*.py` |
| Figure 3 多种子 PR 曲线 / Figure 4 区域证据 | `15_figures_and_predictions/` | `build_pr_and_case_materials.py`<br>`export_uniform_region_heatmaps.py`<br>`render_exact_region_evidence.py` | `run_locked_paper_figure_materials.sh` |

各目录内 `protocol_*.txt` 为对应实验的原始协议记录（种子、划分 SHA256、checkpoint 选择规则等）。

---

## 7. 复现指南

### 7.1 数据准备（列表格式、目录布局与路径修正）

**① 标注列表格式**（`data_splits/` 已提供，每行三段）：

```text
<viewA 图像路径> <viewB 图像路径> <K 个多标签，逗号或空格分隔>
```

真实示例（DvXray）：

```text
/home/hfuu/桌面/convnextv2/data/DvXray_Positive_Samples/P03152_OL.png .../P03152_SD.png 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
```

- `K` = `--num_classes`：DvXray 为 15、LDXray 为 12；
- 标签顺序必须与 `classes.txt` 行顺序一致（DvXray：`Gun, Knife, Wrench, Pliers, Scissors, Lighter, Battery, Bat, Razor_blade, Saw_blade, Fireworks, Hammer, Screwdriver, Dart, Pressure_vessel`）；
- 图像路径可以是绝对路径，也可以是相对 `core/` 的相对路径。

**② 划分规模（与论文 4.1.1 节一致）**

| 数据集 | train | val | test | 类别数 |
|---|---|---|---|---|
| DvXray | 12,800 | 1,600 | 1,600 | 15 |
| LDXray | 99,133 | 11,015 | 36,849 | 12 |

**③ 需要你自备的数据目录**（图像本体不随仓库发布）：双视角成对命名即可，例如
DvXray 同目录下 `P03152_OL.png` / `P03152_SD.png`，LDXray 的 `train_A/000000.jpg` / `train_B/000000.jpg`。

**④ 展平到脚本期望的位置**

```bash
bash main.sh setup      # → core/annotations/{classes.txt,DvXray_*.txt} 与 core/annotations/ldxray/
```

**⑤ 修正列表里的路径**（必做，否则训练会在 DataLoader 阶段报找不到图像）：
划分文件保存的是原机器的绝对路径，两种改法任选：

```bash
# 推荐：按前缀批量替换（自动备份为 *.txt.orig）
RELINK_FROM=/home/hfuu/桌面/convnextv2/data  RELINK_TO=/data/DvXray bash main.sh relink
RELINK_FROM=/home/hfuu/桌面/LDXRAY-20260901/dataset_clean  RELINK_TO=/data/LDXray bash main.sh relink

# 或手动：sed -i 's#原前缀#新前缀#g' core/annotations/DvXray_*.txt
```

`preflight` 会抽样检查前 20 行的图像是否真实存在，路径不对会直接报错并提示 relink。

### 7.2 一键复现主结果（main.sh）

`main.sh` 把「准备标注 → 自检 → 训练 → 锁定 Test 评估」串成一条链路，并内置论文协议约束。

| 子命令 | 作用 |
|---|---|
| `bash main.sh` / `all` | `setup → preflight → train → eval → status` |
| `setup` | 把 `data_splits/` 展开到 `core/annotations/` |
| `relink` | 按前缀修正列表里的图像路径（需 `RELINK_FROM` / `RELINK_TO`） |
| `preflight` | 依赖与 CUDA 探测 + 锁定源码/划分 SHA256 校验 + 数据集可访问性检查 |
| `train` | 训练主方法（Val 选点，绝不碰 Test） |
| `eval` | 校验 checkpoint 协议后对 Test 锁定评估一次 |
| `status` | 打印 checkpoint、指标与协议文件位置 |
| `help` | 帮助 |

常用环境变量（全部可覆盖）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PYTHON_BIN` | `/home/hfuu/miniforge3/envs/v2b384_env/bin/python` | 解释器路径 |
| `GPU_ID` | `0` | 写进 `CUDA_VISIBLE_DEVICES` |
| `SEED` | `930163947` | 随机种子 |
| `BATCH_SIZE` / `EPOCHS` / `PATIENCE` | `32` / `180` / `25` | 训练超参 |
| `NUM_WORKERS` | `8` | DataLoader 进程数 |
| `RUN_ID` / `OUT_ROOT` | `run_<日期>_uniform_fusion_main` / `runs_uniform_fusion/<RUN_ID>` | 产物位置 |
| `RESUME_PARTIAL` | `true` | 存在 `checkpoint_last.pth` 时续训 |
| `DRY_RUN` | `false` | 只打印命令不执行 |
| `SKIP_CHECKSUM` / `SKIP_DATA_CHECK` | `false` | 跳过 SHA256 / 数据集检查（结果不再具备协议效力） |
| `FORCE_RETEST` | `false` | 允许重复跑已经评估过的 Test |
| `EXTRA_TRAIN_ARGS` | 空 | 追加任意 `main_finetune.py` 参数 |

常见用法：

```bash
DRY_RUN=true bash main.sh                      # 先看将执行什么
GPU_ID=1 SEED=553800223 bash main.sh all       # 换卡换种子
RUN_ID=my_repro bash main.sh all               # 指定运行标识

# 论文 Table 1 的 n=5 固定种子（禁止替换）
for S in 207027553 553800223 930163947 1716854429 1786430941; do
  SEED=$S RUN_ID=uf_main_n5 bash main.sh all
done
```

> `eval` 会在 `OUT_DIR` 写入 `test_eval_done.marker` 并记录 checkpoint 与 test 列表的 SHA256；
> 重复评估同一配置会被拒绝（除非 `FORCE_RETEST=true`）。这正是论文「Test 只评估一次」的实现方式。

### 7.3 目录布局（脚本对运行目录的假设）

脚本假定「代码根目录 = 当前目录」：

```
<代码根>/main_finetune.py
<代码根>/models/...
<代码根>/tools/evaluate_project_checkpoint.py
<代码根>/annotations/classes.txt          ← 由 data_splits/ 提供
<代码根>/annotations/DvXray_train.txt ...
```

请以 `core/` 作为代码根，并把 `data_splits/DvXray/*`、`data_splits/LDXray/*`
按脚本期望的文件名放入 `core/annotations/`：

```bash
cd /home/hfuu/桌面/UniformFusion_code_submission
mkdir -p core/annotations
cp data_splits/DvXray/* core/annotations/            # classes.txt + DvXray_{train,val,test}.txt
cp data_splits/LDXray/* core/annotations/            # LDXray_{train,val,test}.txt + ldxray_classes.txt
```

### 7.4 Uniform Fusion 训练命令（主方法，摘自锁定脚本原文；等价于 `bash main.sh train`）

```bash
python -u main_finetune.py \
  --aug_mode conditional --patience 25 \
  --model resnet50 --model_prefix "" \
  --batch_size 32 --epochs 180 --lr 1e-4 \
  --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2 \
  --input_size 224 --dual_view true --view_mode paired --teacher_mode false \
  --num_workers 8 --seed 930163947 --device cuda \
  --deterministic true --reseed_before_training true \
  --train_list annotations/DvXray_train.txt --val_list annotations/DvXray_val.txt \
  --classes_file annotations/classes.txt --num_classes 15 \
  --fpn_out_channels 256 --gspf_lambda_consistency 0.0 --gspf_lambda_ortho 0.0 \
  --head_type c5 --fuse_mode add \
  --base_loss bce --use_semantic_branch false \
  --summary_csv <SUMMARY_CSV> --output_dir <OUT_DIR> \
  --return_intermediate true --use_p9_caprs true \
  --plain_innovation_levels C4 C5 \
  --plain_innovation_projection_dim 64 --plain_innovation_topk 8 \
  --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1 \
  --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 \
  --plain_innovation_base_floor 0.0 \
  --plain_innovation_use_counterfactual_experts true \
  --plain_innovation_use_learned_router false \
  --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10 \
  --plain_innovation_aux_weight 0.03 --plain_innovation_route_weight 0.0 \
  --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02
```

- **Plain-BCE 基线**：完全相同的命令，但省略 `--return_intermediate` 及其后全部 `--plain_innovation_*`。
- **LDXray（Table 8，StepMatched）**：`--epochs 80 --patience 15`，
  `--plain_innovation_warmup_epochs 3 --plain_innovation_ramp_epochs 5`，
  `--plain_innovation_gamma_init 0.003 --plain_innovation_gamma_max 0.02`，
  `--plain_innovation_guard_weight 0.15`，并关闭 DvXray 的弱类条件增强策略。
- 脚本统一设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 以配合 `--deterministic true`。

### 7.5 评估（Val 选点 → Test 锁定评估一次；等价于 `bash main.sh eval`）

```bash
python -u tools/evaluate_project_checkpoint.py \
  --checkpoint <checkpoint_best.pth> --list annotations/DvXray_test.txt \
  --classes-file annotations/classes.txt --view-mode paired \
  --output-json test_metrics.json --output-csv test_metrics.csv \
  --batch-size 32 --num-workers 8 --device cuda

python tools/verify_checkpoint_protocol.py \
  --checkpoint <checkpoint_best.pth> --expected-val-list annotations/DvXray_val.txt
```

### 7.6 整组实验与调用关系

⚠️ 所有 `run_*.sh` 在原工程中位于**同一目录**，脚本之间用相对文件名互相调用。
本包按论文条目分组只是为了查阅；直接运行时只需把相关 **`.sh`** 放到同一目录（`core/`）即可——
它们引用的 `tools/*.py` 已全部就位于 `core/tools/`。例如跑 Table 5：

```bash
cd core
cp ../experiments/00_shared_runners/*.sh ../experiments/09_controls/*.sh .
TRAIN_LIST=annotations/DvXray_train.txt VAL_LIST=annotations/DvXray_val.txt \
TEST_LIST=annotations/DvXray_test.txt GPU_ID=0 bash run_reviewer_first_batch_9runs.sh
```

调用关系速查：

| 组脚本 | 调用的运行器 | 运行器所在组 |
|---|---|---|
| `01/run_uniform_missing_2seeds.sh` | `run_p9_final_noanchor_ablation_one.sh` | `00_shared_runners` |
| `05/run_uniform_component_ablation_3seeds.sh` | `run_uniform_component_ablation_one.sh` | 本组 |
| `09/run_*`、`10/run_*` | `run_reviewer_first_batch_one.sh` | `00_shared_runners` |
| `01/run_final_noanchor_*.sh` | `run_final_noanchor_locked_pair_one.sh` | 本组 |
| `02`、`03`、`04`、`06`、`08` 的组脚本 | 各自的 `*_one.sh` | 本组 |
| `01/run_all_final_noanchor_evidence.sh` | 依次调用 `01/run_final_noanchor_resnet50_n5.sh`、`03/run_final_noanchor_convnextv2_3seeds.sh`、`12/run_final_noanchor_view_ablation_test_n5.sh`、`14/run_final_noanchor_efficiency.sh` | 跨 4 组，需合并到同目录 |

脚本保留原机器的绝对路径默认值（`PYTHON_BIN=/home/hfuu/miniforge3/envs/v2b384_env/bin/python`）；
`PYTHON_BIN`、`TRAIN_LIST`、`VAL_LIST`、`TEST_LIST`、`SAVE_ROOT`、`GPU_ID`、`BATCH_SIZE` 等
均可通过环境变量覆盖。

### 7.7 实验专用代码快照

部分实验使用与 `core/` **不同的**同名文件版本，复现时需覆盖到 `core/` 对应路径：

| 目录 | 覆盖文件 | 对应实验 |
|---|---|---|
| `02_LDXray_cross_dataset/snapshot/` | `datasets.py`、`engine_finetune.py`、`main_finetune.py`、`utils.py`、`models/convnextv2_dual.py`、`models/tv_backbones.py`、`models/modules/plain_bce_innovations.py`、`tools/evaluate_project_checkpoint.py`、`tools/prepare_ldxray_multilabel.py` | Table 8 LDXray |
| `07_fusion_baselines/snapshot/` | `main_finetune.py`、`models/convnextv2_dual.py`、`tools/{archive_fair_fusion_baselines,evaluate_project_checkpoint,verify_checkpoint_protocol}.py` | Table 2 融合基线 |
| `08_adapter_baselines/` | `main_finetune.py`、`models/fair_baseline_adapters.py`（含 `--accum_iter` 梯度累积） | Table 3 适配器 |
| `15_figures_and_predictions/snapshot/` | `datasets.py`、`tools/{export_locked_predictions,locked_checkpoint_utils,materialize_sample_predictions_csv}.py` | Fig.3/Fig.4 样本级导出 |

### 7.8 产物与目录结构

`bash main.sh all` 完成后（`OUT_ROOT` 默认 `./runs_uniform_fusion/<RUN_ID>`）：

```text
runs_uniform_fusion/<RUN_ID>/
├── main.log                      完整 stdout（训练 + 评估，脚本自动 tee）
├── protocol.txt                  本次运行的协议声明与各列表 SHA256
├── training_results.csv          Val 每个 epoch 的指标（checkpoint 选择依据）
└── resnet50/seed_<SEED>/
    ├── checkpoint_best.pth       ★ Val 最佳 checkpoint（Test 评估使用）
    ├── checkpoint_last.pth       最后一个 epoch（供 RESUME_PARTIAL 续训）
    ├── training_log.csv          按 mAP 排序的训练日志
    ├── test_metrics.json         ★ 锁定 Test 指标（论文表格数据来源）
    ├── test_metrics.csv          逐类 AP 明细
    └── test_eval_done.marker     Test 已评估锁 + checkpoint/列表 SHA256
```

---

## 8. 协议要点（对应论文 4.1.3 节）

- 训练集训练；**Val 上选最佳 checkpoint**；锁定后 **Test 只评估一次**。
- DvXray 固定五种子：`207027553`、`553800223`、`930163947`、`1716854429`、`1786430941`；
  禁止删除或替换种子。
- LDXray 匹配种子：`930163947`、`1786430941`、`553800223`；LDXray 第一种子早于其余种子评估 Test，
  论文已如实披露该协议偏差。
- `--deterministic true --reseed_before_training true`，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 敏感性实验与视角消融**只报告 Validation**，不评估 Test，以保持预先声明的协议。
- 采用早停，因此相同 max epoch / patience 下各次运行的实际训练轮数不同（论文 4.1.2 节已报告）。
- Figure 4 的热图表示模型内部 class-query Top-K 区域选择权重，**不是** Grad-CAM，也不是人工标注框。

---

## 9. 第三方代码

`experiments/08_adapter_baselines/third_party/` 为上游官方实现的原样快照，仅作对照基线使用，
版权与许可归原作者所有：

- `DAGNet/`（`model/model_v2.py` 与 `module/*.py`）
- `ML_Decoder/ml_decoder.py`

对应官方代码的锁定 commit 见 `third_party_locked_commits.json`。
`core/models/official_ahcr_adapter.py` 与 `core/models/fair_baseline_adapters.py` 为本项目侧适配层。

---

## 10. 校验

```bash
cd /home/hfuu/桌面/UniformFusion_code_submission

# 全包文件级校验（184 项，除清单自身外）
sha256sum -c SHA256SUMS.txt

# 论文协议锁定的源码与划分（等价于 main.sh preflight 中的校验步骤）
sha256sum core/main_finetune.py core/models/convnextv2_dual.py core/engine_finetune.py
sha256sum data_splits/DvXray/DvXray_train.txt data_splits/DvXray/DvXray_val.txt data_splits/DvXray/DvXray_test.txt
```

`SHA256SUMS.txt` 覆盖论文归档的 184 个文件；校验值与论文各实验 `protocol.txt` 记录一致，
是「跑的是论文那一版代码」的凭据。其中仅 `README.md` 一项因本文档为公开发布重写而更新了哈希，
其余 183 项与归档记录逐字节一致。

`SHA256SUMS.restored.txt` 记录本次公开发布新增的 24 个辅助工具脚本的哈希，同样可校验：

```bash
sha256sum -c SHA256SUMS.restored.txt
```

发布后新增、不在冻结清单内的文件：`main.sh`、`LICENSE`、`CITATION.cff`、`requirements.txt`、
`environment_versions.txt`、`.gitignore`、`SHA256SUMS.restored.txt`。

---

## 11. 已知限制

**① 辅助工具脚本已从归档补齐。** 原冻结包缺少 24 个被组脚本引用的工具（smoke 自检、
结果汇总、图表构建、AHCR 严格协议适配等），本次发布已从论文归档
（`论文最终归档_20260830/05_运行配置与代码/tools/` 与原始工程 `tools/` 快照）恢复至 `core/tools/`，
并已确认:

- 24 个文件全部通过语法编译检查，且被引用的工具链无未解析依赖；
- 其中 10 个在原包中已有副本（位于对应实验组目录），经哈希比对与恢复版**逐字节一致**；
- 哈希记录在 `SHA256SUMS.restored.txt`（不属于论文冻结清单 `SHA256SUMS.txt`）。

因此原先受影响的 14 个组脚本现在可以完整执行（自检 + 训练 + 汇总）。

**② 组脚本需合并到同一目录运行**（见 §7.6）：原工程中所有 `run_*.sh` 同处一个目录，
彼此以相对文件名调用；本仓库按论文章节分组只是为了便于查阅。

**③ 划分清单里是原机器绝对路径**，必须按 §7.1 修正后才能训练。

**④ LDXray 存在一处已披露的协议偏差**：第一个种子早于其余种子评估 Test，论文已如实说明。

---

## 12. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `数据集路径检查未通过` | 划分文件里是原机器的绝对路径 → 按 §7.1 执行 `main.sh relink` |
| `checkpoint selection split mismatch` | 评估用的 `VAL_LIST` 与训练时不一致；两者必须等价（同路径、同内容） |
| `该 checkpoint 已评估过 Test` | 协议保护生效；确需重跑设 `FORCE_RETEST=true` |
| `CUDA out of memory` | 降 `BATCH_SIZE`（16 / 8），或改 `GPU_ID`；确认没有其他 `main_finetune.py` 在跑 |
| 训练完却没有 `checkpoint_best.pth` | `EPOCHS` 太少或 Val 指标始终未提升；按论文用 `EPOCHS=180 PATIENCE=25` |
| 不同次运行的 epoch 数不一样 | 早停导致，论文 4.1.2 节已说明；固定种子 + `--deterministic` 保证同种子可复现 |
| 组脚本报 `tools/xxx.py` 不存在 | 该工具未在 `core/tools/`；按 §11 从归档恢复或拷贝到 `core/tools/` |
| 需要完全离线 | 脚本默认 `HF_HUB_OFFLINE=1`；若主干需本地权重请加 `--model_source local` 并备好权重文件 |

---

## 13. 引用与许可

- **代码仓库**：https://github.com/cangyiyiyiyi-creator/UniformFusion
- **已发布版本**：**v1.0-paper**（附注 tag）。要获取与论文一致的锁定快照：

  ```bash
  git clone --branch v1.0-paper https://github.com/cangyiyiyiyi-creator/UniformFusion.git
  ```

- **引用元数据**：`CITATION.cff` —— GitHub 仓库页右上角的「Cite this repository」按钮即由此生成。

对应稿件：*Class-Conditioned Multi-Scale Regional Learning for Dual-View X-Ray Multi-Label Recognition*
（`UniformFusion_submission_revision_20260919_tablebold.pdf`）。正式发表后请补充期刊/会议与 DOI。

软件条目（可在论文与 README 中直接引用）：

```bibtex
@software{uniformfusion2026,
  title     = {UniformFusion: Class-Conditioned Multi-Scale Regional Learning
               for Dual-View X-Ray Multi-Label Recognition},
  author    = {代程宇},
  year      = {2026},
  version   = {v1.0-paper},
  license   = {MIT},
  url       = {https://github.com/cangyiyiyiyi-creator/UniformFusion}
}
```

- 本仓库采用 **MIT License**（见 `LICENSE`），版权归 2026 代程宇；覆盖 `core/`、`experiments/`、`main.sh`。
- `experiments/08_adapter_baselines/third_party/` 为上游官方实现的原样快照（DAGNet、ML-Decoder），
  版权与许可归原作者所有（不适用上述 MIT 条款），锁定 commit 见 `third_party_locked_commits.json`。
- DvXray / LDXray 数据集请遵循各自原始许可；本仓库仅提供划分清单，不含图像。
