# UniformFusion — Official Code for the Paper

> Class-Conditioned Multi-Scale Regional Learning for Dual-View X-Ray Multi-Label Recognition
> (manuscript: `UniformFusion_submission_revision_20260919_tablebold.pdf`)

This repository is the official implementation and reproduction code of the paper above. It contains the
main method **Uniform Fusion**, the locked sources of every comparison experiment, the run scripts and the
data split manifests. All code comes from a frozen snapshot of the paper archive, with SHA256 hashes
matching the values recorded in each experiment's `protocol.txt`.

| | |
|---|---|
| **Task** | Dual-view (OL / SD) X-ray prohibited-item multi-label recognition |
| **Main method** | `resnet50` dual branch + C4/C5 multi-scale class-query Top-K regional evidence + counterfactual experts + uniform aggregation (internal config name `Final_NoAnchor_NoRouter`) |
| **Datasets** | DvXray (15 classes, 12,800 / 1,600 / 1,600), LDXray (12 classes, 99,133 / 11,015 / 36,849) |
| **Environment** | Python 3.10.20 / PyTorch 2.9.0+cu128 / timm 1.0.22 |
| **Entry point** | `bash main.sh` (prepare data links → preflight → training → locked test evaluation) |

---

## Quick start (5 steps)

```bash
# 0) Environment (see section 3)
conda create -n v2b384_env python=3.10.20 -y && conda activate v2b384_env
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1) Prepare annotations (expand data_splits/ into the core/annotations/ layout the scripts expect)
bash main.sh setup

# 2) Point the split files at your own data root (see section 7.1)
#    first inspect the stored prefix:  head -1 data_splits/DvXray/DvXray_train.txt
RELINK_FROM='<original prefix>' RELINK_TO=/data/DvXray bash main.sh relink

# 3) Preflight (interpreter / dependencies / CUDA, locked-source and split SHA256, dataset accessibility)
bash main.sh preflight

# 4) Train the main method (train on train, select the checkpoint on validation)
GPU_ID=0 SEED=930163947 bash main.sh train

# 5) Locked test evaluation, performed exactly once per configuration
bash main.sh eval
```

To see what would be executed without running it: `DRY_RUN=true bash main.sh`.
Subcommands are documented in section 7.2, per-table reproduction commands in sections 6 and 7.6,
and the output layout in section 7.8.

**Scope**: this package only ships the code that corresponds to experiments actually reported in the
manuscript. Methods not mentioned in the paper (semantic / LLM branch, discriminative centering, the
SIXray and CUB cross-dataset runs, the p16–p20 candidate-head ablations, learned-router robustness,
best-route selection, etc.) are deliberately excluded.

**Provenance**: everything is taken from the frozen snapshot of the paper archive
`paper_archive_20260830/`.

---

## 1. Directory layout

```
UniformFusion/
├── README.md                     this file (full reproduction instructions)
├── main.sh                       ★ one-stop entry: setup / relink / preflight / train / eval / status
├── LICENSE                       MIT License (except the `third_party/` snapshots, see §13)
├── CITATION.cff                  citation metadata (drives GitHub's "Cite this repository" button)
├── environment_versions.txt      recorded environment (conda env v2b384_env)
├── requirements.txt              core dependencies (full 81-package list in pip_freeze.txt)
├── pip_freeze.txt                complete pip freeze of the environment
├── SHA256SUMS.txt                manifest of this release (216 files, English edition)
├── SHA256SUMS.restored.txt       hashes of the 24 helper tools restored from the archive
├── HASH_MAPPING.txt              original (v1.0-paper) ↔ current (v1.1-paper) hashes of the 57 translated files
├── .gitignore                    ignores run outputs / weights / caches
│
├── core/                         ★ locked sources of the paper (runnable as-is)
│   ├── main_finetune.py          training entry point (354 arguments)
│   ├── engine_finetune.py        training / evaluation engine
│   ├── datasets.py               dual-view dataset and conditional augmentation policy
│   ├── utils.py, optim_factory.py
│   ├── models/
│   │   ├── convnextv2_dual.py                ★ main model
│   │   ├── modules/plain_bce_innovations.py  ★ regional evidence / experts / Guard / uniform aggregation
│   │   ├── modules/visual_evidence.py        C4/C5 multi-scale class-query Top-K selection
│   │   ├── modules/{fusions,necks,attentions,cross_view_consistency,augmentations,common,losses,composite_loss}.py
│   │   ├── modules/plain_bce_p16..p20.py     imported at the top level of convnextv2_dual.py, keep them
│   │   ├── modules/custom_losses/, modules/distillation/
│   │   ├── timm_backbones.py, tv_backbones.py, convnextv1.py, convnextv2.py, utils.py
│   │   └── official_ahcr_adapter.py, fair_baseline_adapters.py
│   └── tools/                    all 39 helper scripts: evaluation, protocol verification, statistics,
│                                 profiling, smoke tests, summarisation, figure building
│                                 (including the 24 restored from the archive, see §11)
│
├── data_splits/                  split lists (plain text paths, no images)
│   ├── DvXray/{classes.txt, DvXray_train.txt, DvXray_val.txt, DvXray_test.txt}
│   └── LDXray/{ldxray_classes.txt, LDXray_train.txt, LDXray_val.txt, LDXray_test.txt, LDXray_split_manifest.json}
│
└── experiments/                  run scripts grouped by paper item + experiment-specific snapshots
    ├── 00_shared_runners/        single-run runners shared by several groups
    ├── 01_DvXray_main_n5/                 Table 1
    ├── 02_LDXray_cross_dataset/           Table 8
    ├── 03_ConvNeXtV2_backbone/            §4.2.3
    ├── 04_MaxViT_backbone/                §4.2.3
    ├── 05_component_analysis/             Table 4
    ├── 06_AHCR_comparison/                §4.2.2
    ├── 07_fusion_baselines/               Table 2
    ├── 08_adapter_baselines/              Table 3
    ├── 09_controls/                       Table 5
    ├── 10_correction_and_sensitivity/     Table 6 + §4.3.4 sensitivity
    ├── 11_single_view/                    Table 7
    ├── 12_view_ablation/                  §4.3.4 view mismatch
    ├── 13_internal_vs_final_analysis/     §4.3.3
    ├── 14_efficiency/                     Table 9
    └── 15_figures_and_predictions/        Figure 3 / Figure 4
```

---

## 2. What is not included

| Missing item | Notes |
|---|---|
| Dataset images | the X-ray images of DvXray and LDXray |
| Dataset annotations | LDXray raw JSON; DvXray raw annotations |
| Pretrained weights | e.g. ConvNeXtV2-Tiny `student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors` |
| Training artefacts | checkpoints, `*_metrics.json/csv`, `train.log`, summary CSVs |
| Paper figures | PDF/PNG/SVG/heat maps and all result tables |
| Image paths inside the split files | the lists store the original machine's absolute paths (`/home/hfuu/...`). The code does not rewrite paths; use `main.sh relink` (§7.1) or edit them yourself to point at your data root |
| Helper scripts | 24 smoke-test / summarisation / figure-building tools were absent from the original frozen package and have been restored into `core/tools/` from the paper archive for this public release (see §11) |

> `data_splits/` only provides the **split manifests** (one image path per line) so that you can confirm the
> splits match the paper; it contains no images.

---

## 3. Environment

### 3.1 Creating the environment from scratch (recommended)

```bash
conda create -n v2b384_env python=3.10.20 -y
conda activate v2b384_env

# PyTorch wheels must come from the cu128 index (the paper used 2.9.0+cu128)
pip install torch==2.9.0 torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Remaining core dependencies
pip install -r requirements.txt
```

### 3.2 Reusing an existing environment

```bash
conda activate v2b384_env        # Python 3.10.20 / torch 2.9.0+cu128 / timm 1.0.22
pip install -r requirements.txt  # full 81-package list in pip_freeze.txt
```

### 3.3 Measurement environment reported in the paper (section 4.6)

Ubuntu 24.04.4 LTS, Python 3.10.20, PyTorch 2.9.0+cu128, CUDA Runtime 12.8, cuDNN 9.10.2,
NVIDIA GeForce RTX 5080 (16 GB). A machine-readable version is in `environment_versions.txt`.

- A single GPU is enough for every experiment; use `GPU_ID=<n>` to pick one.
- Peak memory of one training run is roughly 6–8 GB (`--batch_size 32`, `resnet50`, `224²`, dual view).
- Optional dependencies (`try/except` or lazy imports; the main experiments run without them):
  `tensorboardX`, `wandb`, `thop`, `apex`, `MinkowskiEngine`.
- For offline machines set `HF_HUB_OFFLINE=1`; `main.sh` already exports
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTORCH_ALLOC_CONF=expandable_segments:True`.

---

## 4. Locked code versions

The three files below are the only authoritative versions behind the reported results. The values in this
table are those of the **English edition** (`v1.1-paper`), i.e. after the comment/string translation.
The original Chinese snapshot is preserved by the **`v1.0-paper` tag**, and `HASH_MAPPING.txt` maps every
original hash to its current value, so the link to the paper's `protocol.txt` stays auditable.

| File | SHA256 (v1.1-paper) | SHA256 (v1.0-paper, as recorded in the paper's `protocol.txt`) |
|---|---|---|
| `core/main_finetune.py` | `e157ce6f264e9b2e8fa841bdc5958863a506966964850ceae961c735f963c2d0` | `d1ec51b1e44152db52def599d2185d02318786de2db1177bff3158b26ce1782d` |
| `core/models/convnextv2_dual.py` | `2ee664f54bc6cd7d6db0e09a40be70f6143072a9eeeb59fc311ff79091426a99` | `f39a507d877dd8350fe862fdfc013032f7c872e6ca869371f9bbdb64fa81abcb` |
| `core/engine_finetune.py` | `3c41cc4b264b19ad26449ce30519451c6e8f4c7c6e54a81c735c188176a86cf7` | `dbe37f44aee890644a3016012b1b240a18ddd933e09fadb18d0ea38e7c1140ff` |

The translation touched **comments, docstrings and printed messages only**: identifiers, argument names,
default values, numerical constants and control flow are unchanged. Evidence: a full smoke run on a
small subset reproduces the pre-translation metrics exactly (`val mAP 0.0280`, `test mAP 0.0218518525`),
and `bash main.sh preflight` verifies these hashes on every run.

Data split verification (identical to the constants hard-coded in the scripts):

| File | SHA256 |
|---|---|
| `data_splits/DvXray/DvXray_train.txt` | `f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf` |
| `data_splits/DvXray/DvXray_val.txt` | `a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67` |
| `data_splits/DvXray/DvXray_test.txt` | `6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6` |

Split sizes match section 4.1.1 of the paper: DvXray 12,800 / 1,600 / 1,600 (15 classes);
LDXray 99,133 / 11,015 / 36,849 (12 classes, its validation split is a 10% sample of the official
training set drawn with seed 20260901).

---

## 5. Method naming (read this first)

| Name in the paper | Internal configuration name | Key flags |
|---|---|---|
| **Uniform Fusion** (main method) | `Final_NoAnchor_NoRouter` | `--plain_innovation_use_learned_router false` and `--plain_innovation_route_weight 0.0` |
| Learned Router (historical variant, not used in the paper) | `Final_NoAnchor` | `use_learned_router true`, `route_weight 0.05` |

Evidence from the result archive (`04_original_result_tables/06_uniform_fusion_main_n5/README.md`, quoted verbatim):
"`Final_NoAnchor` in these tables denotes the historical, complete Learned Router variant;
`Uniform_Fusion` is the current main method."

Code-level evidence: in `core/models/modules/plain_bce_innovations.py`, when `use_learned_router=False`
the expert weights are taken directly from `paired.new_full(..., 1.0 / expert_count)`, i.e. uniform
weights — exactly the "Uniform logit aggregation" described in section 3.4 of the paper.

> ⚠️ `run_final_noanchor_locked_pair_one.sh` supports both `Plain_BCE` and `Final_NoAnchor`.
> It is kept here because the **n=5 Plain_BCE baseline of Table 1 was produced by it**;
> its `METHOD=Final_NoAnchor` branch is not part of the paper — do not use it.

---

## 6. Experiment → paper mapping

| Paper item | Directory | Key scripts | Runners used |
|---|---|---|---|
| Table 1 DvXray ResNet-50 n=5 | `01_DvXray_main_n5/` | `run_uniform_missing_2seeds.sh` (completes the main method to n=5)<br>`run_final_noanchor_resnet50_n5.sh` (Plain-BCE baseline) | `00_shared_runners/run_p9_final_noanchor_ablation_one.sh`<br>`01/run_final_noanchor_locked_pair_one.sh` |
| Table 2 fusion configurations (Mean/Max/Concat/Cross-attn) | `07_fusion_baselines/` | `run_fair_fusion_baselines_3seeds.sh` | same directory |
| Table 3 DAGNet / ML-Decoder adapters | `08_adapter_baselines/` | `run_fair_adapters_3seeds.sh`<br>`run_dagnet_batch32_recheck_3seeds.sh` | `08/run_fair_adapter_one.sh` |
| §4.2.2 AHCR under both protocols, n=5 | `06_AHCR_comparison/` | `run_ahcr_official_strict_5seeds.sh`<br>`run_official_ahcr_resnet50_5seeds.sh` | `06/run_ahcr_official_strict_one.sh`, `tools/run_ahcr_official_strict.py` |
| §4.2.3 ConvNeXtV2-Tiny, n=3 | `03_ConvNeXtV2_backbone/` | `run_convnextv2_uniform_fusion_3seeds.sh` | `03/run_convnextv2_uniform_fusion_one.sh` |
| §4.2.3 MaxViT-Tiny, n=3 | `04_MaxViT_backbone/` | `run_maxvit_uniform_3seeds.sh` | `04/run_maxvit_uniform_one.sh` |
| Table 4 component analysis (no duplicated-view branch / no guard / C5 only) | `05_component_analysis/` | `run_uniform_component_ablation_3seeds.sh` | `05/run_uniform_component_ablation_one.sh` |
| Table 5 supervision and spatial-selection controls (incl. GAP / dense / NoAux) | `09_controls/` | `run_reviewer_first_batch_9runs.sh`<br>`run_uf_gap_3seeds.sh`, `run_uf_noaux_3seeds.sh` | `00_shared_runners/run_reviewer_first_batch_one.sh` |
| Table 6 / §4.3.4 correction path and sensitivity | `10_correction_and_sensitivity/` | `run_uniform_residual_ablation_3seeds.sh` (NoCorrection/NoRamp)<br>`run_uniform_val_sensitivity.sh` (K/τ/γmax) | as above |
| Table 7 independent single-view baselines | `11_single_view/` | `run_single_view_plain_3seeds.sh` | same directory |
| §4.3.4 view mismatch (OL copy / SD copy / cyclic mismatch) | `12_view_ablation/` | `run_final_noanchor_view_ablation_test_n5.sh` | reuses the five locked checkpoints from `01/` |
| §4.3.3 internal base branch vs. final residual (training-free) | `13_internal_vs_final_analysis/` | `analyze_uniform_internal_evidence.py` | — |
| Table 8 LDXray n=3 | `02_LDXray_cross_dataset/` | `run_ldxray_uniform_stepmatched_3seeds.sh` | `02/run_ldxray_uniform_one.sh` |
| Table 9 efficiency and cross-session latency | `14_efficiency/` | `run_final_noanchor_efficiency.sh`<br>`run_final_efficiency_5sessions.sh` | `14/tools/profile_*.py` |
| Figure 3 multi-seed PR curves / Figure 4 regional evidence | `15_figures_and_predictions/` | `build_pr_and_case_materials.py`<br>`export_uniform_region_heatmaps.py`<br>`render_exact_region_evidence.py` | `run_locked_paper_figure_materials.sh` |

The `protocol_*.txt` file inside each directory is the original protocol record for that experiment
(seeds, split SHA256, checkpoint-selection rule, etc.).

---

## 7. Reproduction guide

### 7.1 Data preparation (list format, layout, path relinking)

**① Annotation list format** (already provided in `data_splits/`; three fields per line):

```text
<viewA image path> <viewB image path> <K multi-hot labels, comma or space separated>
```

A real example (DvXray):

```text
<DvXray_data_root>/DvXray_Positive_Samples/P03152_OL.png .../P03152_SD.png 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
```

- `K` = `--num_classes`: 15 for DvXray, 12 for LDXray;
- the label order must match the line order of `classes.txt` (DvXray: `Gun, Knife, Wrench, Pliers, Scissors, Lighter, Battery, Bat, Razor_blade, Saw_blade, Fireworks, Hammer, Screwdriver, Dart, Pressure_vessel`);
- image paths may be absolute or relative to `core/`.

**② Split sizes (matching section 4.1.1 of the paper)**

| Dataset | train | val | test | classes |
|---|---|---|---|---|
| DvXray | 12,800 | 1,600 | 1,600 | 15 |
| LDXray | 99,133 | 11,015 | 36,849 | 12 |

**③ Data directories you have to provide** (the images themselves are not distributed): paired naming is
enough, e.g. `P03152_OL.png` / `P03152_SD.png` in the same DvXray directory, or `train_A/000000.jpg` /
`train_B/000000.jpg` for LDXray.

**④ Expand the manifests into the expected location**

```bash
bash main.sh setup      # → core/annotations/{classes.txt,DvXray_*.txt} and core/annotations/ldxray/
```

**⑤ Fix the paths inside the lists** (mandatory, otherwise the DataLoader will not find the images).
The split files store the original machine's absolute paths. Inspect the prefix and rewrite it:

```bash
# 1) look at the prefix stored in the manifests
head -1 core/annotations/DvXray_train.txt

# 2) rewrite that prefix to your own data root (originals are backed up as *.txt.orig)
RELINK_FROM='<original prefix read above>'  RELINK_TO=/data/DvXray bash main.sh relink
RELINK_FROM='<original LDXray prefix>'      RELINK_TO=/data/LDXray bash main.sh relink

# or by hand: sed -i 's#<old prefix>#<new prefix>#g' core/annotations/DvXray_*.txt
```

For the LDXray runners (`experiments/02_LDXray_cross_dataset/*.sh`) the dataset root defaults to the
relative `./data/LDXray`; pass `DATASET_ROOT=/your/path` if your images live elsewhere.

`preflight` samples the first 20 lines of every list and fails with an actionable message when the images
cannot be reached.

### 7.2 One-command reproduction of the main result (main.sh)

`main.sh` chains "prepare annotations → preflight → training → locked test evaluation" and enforces the
paper protocol along the way.

| Subcommand | Purpose |
|---|---|
| `bash main.sh` / `all` | `setup → preflight → train → eval → status` |
| `setup` | expand `data_splits/` into `core/annotations/` |
| `relink` | rewrite image path prefixes in the lists (needs `RELINK_FROM` / `RELINK_TO`) |
| `preflight` | dependency and CUDA probe + locked-source/split SHA256 checks + dataset accessibility check |
| `train` | train the main method (selects on validation, never touches the test set) |
| `eval` | verify the checkpoint protocol, then evaluate the test set exactly once |
| `status` | print the location of checkpoints, metrics and protocol files |
| `help` | usage |

Frequently used environment variables (all overridable):

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | `/home/hfuu/miniforge3/envs/v2b384_env/bin/python` | interpreter path |
| `GPU_ID` | `0` | exported as `CUDA_VISIBLE_DEVICES` |
| `SEED` | `930163947` | random seed |
| `BATCH_SIZE` / `EPOCHS` / `PATIENCE` | `32` / `180` / `25` | training hyper-parameters |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `RUN_ID` / `OUT_ROOT` | `run_<date>_uniform_fusion_main` / `runs_uniform_fusion/<RUN_ID>` | output location |
| `RESUME_PARTIAL` | `true` | resume when `checkpoint_last.pth` exists |
| `DRY_RUN` | `false` | print commands without executing them |
| `SKIP_CHECKSUM` / `SKIP_DATA_CHECK` | `false` | skip the SHA256 / dataset checks (results lose protocol validity) |
| `FORCE_RETEST` | `false` | allow re-running an already evaluated test set |
| `EXTRA_TRAIN_ARGS` | empty | append arbitrary `main_finetune.py` arguments |

Common invocations:

```bash
DRY_RUN=true bash main.sh                      # see what would run
GPU_ID=1 SEED=553800223 bash main.sh all       # different GPU and seed
RUN_ID=my_repro bash main.sh all               # custom run identifier

# the five fixed seeds of Table 1 (do not substitute them)
for S in 207027553 553800223 930163947 1716854429 1786430941; do
  SEED=$S RUN_ID=uf_main_n5 bash main.sh all
done
```

> `eval` writes `test_eval_done.marker` into `OUT_DIR`, recording the SHA256 of the checkpoint and of the
> test list. Re-evaluating the same configuration is refused unless `FORCE_RETEST=true`. This is exactly
> how the paper's "evaluate the test set only once" rule is enforced.

### 7.3 Directory layout (what the scripts assume about the working directory)

The scripts assume "code root == current working directory":

```
<code root>/main_finetune.py
<code root>/models/...
<code root>/tools/evaluate_project_checkpoint.py
<code root>/annotations/classes.txt          ← provided by data_splits/
<code root>/annotations/DvXray_train.txt ...
```

Use `core/` as the code root and place `data_splits/DvXray/*` and `data_splits/LDXray/*`
into `core/annotations/` under the file names the scripts expect:

```bash
cd /path/to/UniformFusion
mkdir -p core/annotations
cp data_splits/DvXray/* core/annotations/            # classes.txt + DvXray_{train,val,test}.txt
cp data_splits/LDXray/* core/annotations/            # LDXray_{train,val,test}.txt + ldxray_classes.txt
```

### 7.4 Uniform Fusion training command (main method, taken verbatim from the locked script; equivalent to `bash main.sh train`)

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

- **Plain-BCE baseline**: exactly the same command, but omit `--return_intermediate` and every
  `--plain_innovation_*` flag that follows it.
- **LDXray (Table 8, StepMatched)**: `--epochs 80 --patience 15`,
  `--plain_innovation_warmup_epochs 3 --plain_innovation_ramp_epochs 5`,
  `--plain_innovation_gamma_init 0.003 --plain_innovation_gamma_max 0.02`,
  `--plain_innovation_guard_weight 0.15`, with the DvXray weak-class conditional augmentation disabled.
- The scripts always export `CUBLAS_WORKSPACE_CONFIG=:4096:8` to support `--deterministic true`.

### 7.5 Evaluation (select on validation → evaluate the test set once; equivalent to `bash main.sh eval`)

```bash
python -u tools/evaluate_project_checkpoint.py \
  --checkpoint <checkpoint_best.pth> --list annotations/DvXray_test.txt \
  --classes-file annotations/classes.txt --view-mode paired \
  --output-json test_metrics.json --output-csv test_metrics.csv \
  --batch-size 32 --num-workers 8 --device cuda

python tools/verify_checkpoint_protocol.py \
  --checkpoint <checkpoint_best.pth> --expected-val-list annotations/DvXray_val.txt
```

### 7.6 Whole-group experiments and their call graph

⚠️ In the original project all `run_*.sh` scripts lived in **one directory** and called each other by
relative file name. The grouping in this repository is only for navigation; to run a group you just need to
copy its **`.sh`** files into the same directory (`core/`) — the `tools/*.py` they reference are already in
place under `core/tools/`. Example: running Table 5.

```bash
cd core
cp ../experiments/00_shared_runners/*.sh ../experiments/09_controls/*.sh .
TRAIN_LIST=annotations/DvXray_train.txt VAL_LIST=annotations/DvXray_val.txt \
TEST_LIST=annotations/DvXray_test.txt GPU_ID=0 bash run_reviewer_first_batch_9runs.sh
```

Call-graph cheat sheet:

| Group script | Runner it calls | Runner location |
|---|---|---|
| `01/run_uniform_missing_2seeds.sh` | `run_p9_final_noanchor_ablation_one.sh` | `00_shared_runners` |
| `05/run_uniform_component_ablation_3seeds.sh` | `run_uniform_component_ablation_one.sh` | same directory |
| `09/run_*`, `10/run_*` | `run_reviewer_first_batch_one.sh` | `00_shared_runners` |
| `01/run_final_noanchor_*.sh` | `run_final_noanchor_locked_pair_one.sh` | same directory |
| group scripts of `02`, `03`, `04`, `06`, `08` | their own `*_one.sh` | same directory |
| `01/run_all_final_noanchor_evidence.sh` | calls `01/run_final_noanchor_resnet50_n5.sh`, `03/run_final_noanchor_convnextv2_3seeds.sh`, `12/run_final_noanchor_view_ablation_test_n5.sh`, `14/run_final_noanchor_efficiency.sh` in order | spans 4 groups, merge them into one directory |

The scripts keep the original machine's absolute defaults
(`PYTHON_BIN=/home/hfuu/miniforge3/envs/v2b384_env/bin/python`); `PYTHON_BIN`, `TRAIN_LIST`, `VAL_LIST`,
`TEST_LIST`, `SAVE_ROOT`, `GPU_ID`, `BATCH_SIZE` and others can all be overridden through environment
variables.

### 7.7 Experiment-specific code snapshots

Some experiments use **different** versions of files that share their name with `core/`. To reproduce them,
overwrite the corresponding path inside `core/`:

| Directory | Files to overwrite | Paper item |
|---|---|---|
| `02_LDXray_cross_dataset/snapshot/` | `datasets.py`, `engine_finetune.py`, `main_finetune.py`, `utils.py`, `models/convnextv2_dual.py`, `models/tv_backbones.py`, `models/modules/plain_bce_innovations.py`, `tools/evaluate_project_checkpoint.py`, `tools/prepare_ldxray_multilabel.py` | Table 8 LDXray |
| `07_fusion_baselines/snapshot/` | `main_finetune.py`, `models/convnextv2_dual.py`, `tools/{archive_fair_fusion_baselines,evaluate_project_checkpoint,verify_checkpoint_protocol}.py` | Table 2 fusion baselines |
| `08_adapter_baselines/` | `main_finetune.py`, `models/fair_baseline_adapters.py` (adds `--accum_iter` gradient accumulation) | Table 3 adapters |
| `15_figures_and_predictions/snapshot/` | `datasets.py`, `tools/{export_locked_predictions,locked_checkpoint_utils,materialize_sample_predictions_csv}.py` | Fig. 3 / Fig. 4 sample-level export |

### 7.8 Outputs and directory layout

After `bash main.sh all` finishes (`OUT_ROOT` defaults to `./runs_uniform_fusion/<RUN_ID>`):

```text
runs_uniform_fusion/<RUN_ID>/
├── main.log                      full stdout (training + evaluation, teed by the script)
├── protocol.txt                  protocol declaration of this run and list SHA256 values
├── training_results.csv          per-epoch validation metrics (basis for checkpoint selection)
└── resnet50/seed_<SEED>/
    ├── checkpoint_best.pth       ★ best checkpoint on validation (used for the test evaluation)
    ├── checkpoint_last.pth       last epoch (for RESUME_PARTIAL)
    ├── training_log.csv          training log sorted by mAP
    ├── test_metrics.json         ★ locked test metrics (the numbers behind the paper's tables)
    ├── test_metrics.csv          per-class AP breakdown
    └── test_eval_done.marker     test-evaluation lock + checkpoint/list SHA256
```

---

## 8. Protocol essentials (section 4.1.3 of the paper)

- Train on the training split; **select the best checkpoint on validation**; after locking, **evaluate the
  test set exactly once**.
- DvXray uses five fixed seeds: `207027553`, `553800223`, `930163947`, `1716854429`, `1786430941`.
  They must not be removed or replaced.
- LDXray matched seeds: `930163947`, `1786430941`, `553800223`. The first LDXray seed was evaluated on the
  test set earlier than the others; the paper discloses this protocol deviation explicitly.
- `--deterministic true --reseed_before_training true`, together with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- The sensitivity experiments and the view ablation **report validation only** and never evaluate the test
  set, in order to keep the pre-declared protocol intact.
- Early stopping is used, so under the same max-epoch / patience setting individual runs train for
  different numbers of epochs (reported in section 4.1.2 of the paper).
- The Figure 4 heat maps visualise the model's internal class-query Top-K region-selection weights.
  They are **not** Grad-CAM and they are **not** human-annotated boxes.

---

## 9. Third-party code

`experiments/08_adapter_baselines/third_party/` contains verbatim snapshots of upstream official
implementations, used only as comparison baselines. Copyright and license remain with their original
authors:

- `DAGNet/` (`model/model_v2.py` and `module/*.py`)
- `ML_Decoder/ml_decoder.py`

The locked upstream commits are listed in `third_party_locked_commits.json`.
`core/models/official_ahcr_adapter.py` and `core/models/fair_baseline_adapters.py` are this project's
own adaptation layers.

---

## 10. Verification

```bash
cd /path/to/UniformFusion

# file-level verification of the whole release
sha256sum -c SHA256SUMS.txt

# the sources and splits locked by the paper protocol (the same check `main.sh preflight` performs)
sha256sum core/main_finetune.py core/models/convnextv2_dual.py core/engine_finetune.py
sha256sum data_splits/DvXray/DvXray_train.txt data_splits/DvXray/DvXray_val.txt data_splits/DvXray/DvXray_test.txt

# hashes of the 24 helper tools restored from the archive
sha256sum -c SHA256SUMS.restored.txt
```

**Note on hashes.** This repository is the English edition of the frozen paper archive: comments,
docstrings and printed messages were translated, which necessarily changes the SHA256 of every file that
contained them. `HASH_MAPPING.txt` therefore records the original hash of each translated file next to its
current hash, so the link to the paper's `protocol.txt` remains auditable. The translation changed no
executable logic: identifiers, default values, argument names and numerical constants are untouched. If
you need the byte-identical original files, use the `v1.0-paper` tag, which still points at the
untranslated snapshot.

Files added on top of the frozen archive: `main.sh`, `LICENSE`, `CITATION.cff`, `requirements.txt`,
`environment_versions.txt`, `.gitignore`, `SHA256SUMS.restored.txt`, `HASH_MAPPING.txt`.

---

## 11. Known limitations

**① Helper scripts have been restored from the archive.** The original frozen package was missing 24 tools
referenced by the group scripts (smoke tests, summarisation, figure building, the strict AHCR protocol
adapter). This release restores them into `core/tools/` from the paper archive
(`paper_archive_20260830/05_run_config_and_code/tools/` and the original project's `tools/` snapshot). Verified:

- all 24 files compile, and the tool chain they reference has no unresolved imports;
- 10 of them already had a copy inside the frozen package (in the corresponding experiment group); those
  copies are **byte-identical** to the restored versions;
- their hashes are also recorded in `SHA256SUMS.restored.txt`;

The 14 group scripts that were affected can now run end to end (preflight + training + summarisation).

**② Group scripts must be merged into one directory** (see §7.6): in the original project all `run_*.sh`
lived in a single directory and called each other by relative file name. The grouping in this repository is
only for navigation.

**③ The split manifests contain the original machine's absolute paths** and must be fixed before training
(§7.1).

**④ One disclosed protocol deviation exists for LDXray**: its first seed was evaluated on the test set
before the remaining seeds; the paper states this explicitly.

**⑤ Archive-side tools expect the paper archive inside the repository.** 31 files — most of
`core/tools/` plus several `experiments/` group scripts (`03`, `05`, `07`, `08`, `09`, `10`, `11`, `13`,
`14`, `15`) — read the paper archive and its supplementary runs, for example
`paper_archive_20260830/05_run_config_and_code/...`. All directory names were translated to English for
this release, so if you have the archive, place (or symlink) it at the repository root:

```text
<repo_root>/
├── paper_archive_20260830/
│   ├── 02_uniform_fusion_main_models/            # main-method checkpoints
│   ├── 03_baseline_and_ablation_models/          # baseline / ablation checkpoints
│   ├── 05_run_config_and_code/                   # the locked sources
│   ├── 06_final_paper_materials_20260831/04_paper_tables/
│   ├── 07_ldxray_cross_dataset_20260902/03_best_models/
│   ├── 08_visualisation_and_pr_curves_20260902/
│   └── 15_fair_fusion_baselines_20260904/
├── supplementary_verification_20260905/
└── supplementary_verification_20260907/
```

The paths are constructed relative to the repository root (`ROOT / "paper_archive_20260830"`), and
`core/tools/analyze_uniform_internal_evidence.py` additionally accepts `--archive <dir>`.

> **`main.sh`, training, evaluation and the checkpoint-protocol check do not need the archive** — the main
> method can be reproduced from this repository alone (§7.2). Only the archive/aggregation, figure-building
> and a few group scripts depend on it.

---

## 12. FAQ

| Symptom | Cause and fix |
|---|---|
| `dataset path check failed` | the split files hold the original machine's absolute paths → run `main.sh relink` as described in §7.1 |
| `checkpoint selection split mismatch` | the `VAL_LIST` used for evaluation differs from the one used for training; both must be equivalent (same path, same content) |
| `this checkpoint has already been evaluated on the test set` | the protocol guard is working; set `FORCE_RETEST=true` if you really need to re-run it |
| `CUDA out of memory` | lower `BATCH_SIZE` (16 / 8) or switch `GPU_ID`; make sure no other `main_finetune.py` is running |
| Training finishes but no `checkpoint_best.pth` appears | `EPOCHS` too small or the validation metric never improved; use the paper setting `EPOCHS=180 PATIENCE=25` |
| Different runs train for different numbers of epochs | caused by early stopping, as explained in section 4.1.2 of the paper; a fixed seed plus `--deterministic` makes same-seed runs reproducible |
| A group script reports `tools/xxx.py` not found | that tool is not in `core/tools/`; restore it from the archive or copy it there (see §11) |
| I need to run completely offline | the scripts already set `HF_HUB_OFFLINE=1`; if your backbone needs local weights add `--model_source local` and provide the weight file |

---

## 13. Citation and license

- **Repository**: https://github.com/cangyiyiyiyi-creator/UniformFusion
- **Released versions**:
  - `v1.1-paper` — this English edition (default branch).
  - `v1.0-paper` — the original, untranslated snapshot, byte-identical to the paper archive.

  ```bash
  git clone --branch v1.1-paper https://github.com/cangyiyiyiyi-creator/UniformFusion.git
  ```

- **Citation metadata**: `CITATION.cff` — it drives the "Cite this repository" button on the GitHub page.

Manuscript: *Class-Conditioned Multi-Scale Regional Learning for Dual-View X-Ray Multi-Label Recognition*
(`UniformFusion_submission_revision_20260919_tablebold.pdf`). Please add the venue and DOI once the paper
is formally published.

- This repository is released under the **MIT License** (see `LICENSE`), copyright 2026 Chengyu Dai;
  it covers `core/`, `experiments/` and `main.sh`.
- `experiments/08_adapter_baselines/third_party/` contains verbatim snapshots of upstream official
  implementations (DAGNet, ML-Decoder). Their copyright and license remain with the original authors and
the MIT terms above do not apply to them; the locked commits are in `third_party_locked_commits.json`.
- The DvXray and LDXray datasets remain under their own original licenses; this repository only ships the
  split manifests and contains no images.
