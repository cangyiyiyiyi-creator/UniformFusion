#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "a6bfc1b1299d28e8226c106a94967287a8e30927"
OFFICIAL_HASHES = {
    "dataset.py": "c062806dc00b953cd35c18c8900b27467aeb030e7113c57b806298ac7178bdd9",
    "get_ap.py": "3b8c0f41376b28b36410a8115500cbf462986acb3c2fb268a0828e0f51a3abc2",
    "loss_func.py": "44d88550d69c9b5d648c52c7c85451091953e45a10462090a4f730beeb195808",
    "model_ResNet.py": "bbc135c43b908cb33278d3f1baee510a4074e0c9b1a252d36cf23e8a973eca85",
    "train.py": "2c3d82b23cf3fc0280ae20f0b04b5ee3edeaee1b78740fc8ea2758dfb462ec65",
    "utils.py": "d2d77bfc7ff6254cdb9cda99c07257a11270f1d3b5080ae692b3513f8a096b0e",
}
OFFICIAL_EPOCHS = 30
OFFICIAL_BATCH_SIZE = 32
OFFICIAL_LR = 0.01
OFFICIAL_GRAD_CLIP = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run untouched AHCR modules with the released training recipe."
    )
    parser.add_argument("--source-dir", default="third_party/DvXray_official")
    parser.add_argument("--train-list", default="annotations/DvXray_train.txt")
    parser.add_argument("--val-list", default="annotations/DvXray_val.txt")
    parser.add_argument("--test-list", default="annotations/DvXray_test.txt")
    parser.add_argument("--classes-file", default="annotations/classes.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=930163947)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def execute_source_module(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_locked_official_modules(source_dir: Path) -> dict[str, types.ModuleType]:
    lock_path = source_dir.parent / "DvXray_official_LOCK.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing source lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("commit") != OFFICIAL_COMMIT:
        raise RuntimeError(f"official commit mismatch in {lock_path}")
    for filename, expected in OFFICIAL_HASHES.items():
        path = source_dir / filename
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"official source hash mismatch for {filename}: "
                f"expected={expected}, actual={actual}"
            )

    modules = {}
    for name in ("utils", "get_ap", "model_ResNet", "dataset", "loss_func", "train"):
        modules[name] = execute_source_module(name, source_dir / f"{name}.py")
    return modules


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != 15:
        raise ValueError(f"AHCR expects 15 classes, found {len(names)}")
    return names


class OfficialDeterministicEvalDataset(Dataset):
    """Deterministic evaluation counterpart to the released training dataset."""

    def __init__(self, list_path: Path, official_utils: types.ModuleType) -> None:
        self.lines = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.official_utils = official_utils

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int):
        fields = self.lines[index].split()
        if len(fields) != 3:
            raise ValueError(f"invalid DvXray annotation line: {self.lines[index]}")
        images = []
        for image_path in fields[:2]:
            with Image.open(image_path) as image:
                image = self.official_utils.cvtColor(
                    image.resize((224, 224), Image.BILINEAR)
                )
                array = np.transpose(
                    self.official_utils.preprocess_input(image), (2, 0, 1)
                )
                images.append(array)
        target = np.asarray([int(value) for value in fields[2].split(",")])
        return images[0], images[1], target


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    repeat: int,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state(),
        "seed": seed,
        "repeat": repeat,
        "source_commit": OFFICIAL_COMMIT,
        "selection_rule": "final_epoch_no_validation_selection",
        "official_recipe": {
            "weights": "ResNet50_Weights.IMAGENET1K_V1",
            "optimizer": "SGD",
            "learning_rate": OFFICIAL_LR,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "epochs": OFFICIAL_EPOCHS,
            "batch_size": OFFICIAL_BATCH_SIZE,
            "drop_last": True,
            "gradient_clip": OFFICIAL_GRAD_CLIP,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_final_epoch(
    model: torch.nn.Module,
    list_path: Path,
    output_json: Path,
    output_csv: Path,
    checkpoint_path: Path,
    classes: list[str],
    modules: dict[str, types.ModuleType],
    device: torch.device,
) -> dict:
    dataset = OfficialDeterministicEvalDataset(list_path, modules["utils"])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=modules["dataset"].DvX_dataset_collate,
    )
    meter = modules["get_ap"].AveragePrecisionMeter()
    meter.reset()
    model.eval()
    for image_ol, image_sd, target in loader:
        image_ol = image_ol.to(device)
        image_sd = image_sd.to(device)
        ol_output, sd_output = model(image_ol, image_sd)
        prediction = modules["utils"].confidence_weighted_view_fusion(
            torch.sigmoid(ol_output), torch.sigmoid(sd_output)
        )
        meter.add(prediction.detach().cpu(), target)

    per_class = [float(value) for value in meter.value()]
    map_value = float(np.mean(per_class))
    payload = {
        "protocol": "AHCR-Official released training recipe with external deterministic evaluation",
        "source_commit": OFFICIAL_COMMIT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": OFFICIAL_EPOCHS - 1,
        "selection_rule": "final_epoch_no_validation_selection",
        "evaluation_list": str(list_path),
        "evaluation_list_sha256": sha256(list_path),
        "evaluation_preprocess": "released resize/cvtColor/preprocess_input without training HSV jitter",
        "evaluation_batch_size": 1,
        "fusion": "verbatim released confidence_weighted_view_fusion at batch_size=1",
        "samples": len(dataset),
        "stats": {
            "mAP": map_value,
            "per_class_ap": per_class,
            "per_class_ap_named": dict(zip(classes, per_class)),
        },
    }
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    row = {"epoch": OFFICIAL_EPOCHS - 1, "mAP": map_value}
    row.update({f"AP_{name}": value for name, value in zip(classes, per_class)})
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(
        f"AHCR_OFFICIAL_STRICT_EVAL_OK split={list_path.name} "
        f"samples={len(dataset)} mAP={map_value:.10f}"
    )
    return payload


def run_preflight(
    modules: dict[str, types.ModuleType],
    device: torch.device,
    train_list: Path,
) -> None:
    set_seed(20260831)
    first_line = train_list.read_text(encoding="utf-8").splitlines(keepends=True)[:1]
    train_sample = modules["dataset"].data_loader(first_line, [224, 224])[0]
    eval_sample = OfficialDeterministicEvalDataset(train_list, modules["utils"])[0]
    for sample in (train_sample, eval_sample):
        if sample[0].shape != (3, 224, 224) or sample[1].shape != (3, 224, 224):
            raise RuntimeError("official AHCR preprocessing returned an invalid shape")
        if sample[2].shape != (15,):
            raise RuntimeError("official AHCR target has an invalid shape")
    model = modules["model_ResNet"].AHCR(num_classes=15).to(device)
    criterion = modules["loss_func"].BCELoss().to(device)
    image_ol = torch.randn(1, 3, 224, 224, device=device)
    image_sd = torch.randn(1, 3, 224, 224, device=device)
    target = torch.zeros(1, 15, device=device)
    ol_output, sd_output = model(image_ol, image_sd)
    loss = criterion(ol_output, sd_output, target)
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=OFFICIAL_LR)
    modules["utils"].clip_gradient(optimizer, OFFICIAL_GRAD_CLIP)
    optimizer.step()
    prediction = modules["utils"].confidence_weighted_view_fusion(
        torch.sigmoid(ol_output.detach()), torch.sigmoid(sd_output.detach())
    )
    if ol_output.shape != (1, 15) or sd_output.shape != (1, 15):
        raise RuntimeError("unexpected official AHCR output shape")
    if prediction.shape != (1, 15) or not torch.isfinite(loss):
        raise RuntimeError("official AHCR preflight produced invalid values")
    with tempfile.TemporaryDirectory(prefix="ahcr_official_strict_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pth.tar"
        save_checkpoint(checkpoint_path, 0, model, optimizer, 20260831, 1)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "AHCR_OFFICIAL_STRICT_PREFLIGHT_OK "
        f"commit={OFFICIAL_COMMIT} params={parameter_count}"
    )


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    modules = load_locked_official_modules(source_dir)
    device = torch.device(args.device)
    if args.preflight_only:
        run_preflight(modules, device, resolve_path(args.train_list))
        return
    if not args.output_dir:
        raise ValueError("--output-dir is required for training")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = resolve_path(args.train_list)
    val_list = resolve_path(args.val_list)
    test_list = resolve_path(args.test_list)
    classes = read_classes(resolve_path(args.classes_file))
    checkpoint_path = output_dir / "checkpoint_final_epoch30.pth.tar"
    complete_marker = output_dir / "suite_complete.marker"
    if (
        complete_marker.is_file()
        and (output_dir / "val_metrics.json").is_file()
        and (output_dir / "test_metrics.json").is_file()
    ):
        print(f"AHCR_OFFICIAL_STRICT_SKIP_COMPLETE output={output_dir}")
        return

    set_seed(args.seed)
    model = modules["model_ResNet"].AHCR(num_classes=15).to(device)
    optimizer = torch.optim.SGD(
        params=filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=OFFICIAL_LR,
    )
    criterion = modules["loss_func"].BCELoss().to(device)
    start_epoch = 0
    if args.resume and checkpoint_path.is_file() and not complete_marker.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        print(f"Resuming strict AHCR from epoch {start_epoch + 1}")

    train_lines = train_list.read_text(encoding="utf-8").splitlines(keepends=True)
    train_loader = DataLoader(
        modules["dataset"].data_loader(train_lines, [224, 224]),
        batch_size=OFFICIAL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=modules["dataset"].DvX_dataset_collate,
    )
    modules["train"].device = device
    modules["train"].grad_clip = OFFICIAL_GRAD_CLIP
    modules["train"].print_freq = 100

    for epoch in range(start_epoch, OFFICIAL_EPOCHS):
        if epoch != 0 and epoch % 10 == 0:
            modules["utils"].adjust_learning_rate(optimizer, 0.1)
        modules["train"].train(train_loader, model, criterion, optimizer, epoch)
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, args.seed, args.repeat
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != OFFICIAL_EPOCHS - 1:
        raise RuntimeError("strict AHCR did not reach the released final epoch")

    for split_name, list_path in (("val", val_list), ("test", test_list)):
        output_json = output_dir / f"{split_name}_metrics.json"
        output_csv = output_dir / f"{split_name}_metrics.csv"
        evaluate_final_epoch(
            model,
            list_path,
            output_json,
            output_csv,
            checkpoint_path,
            classes,
            modules,
            device,
        )

    protocol = {
        "method": "AHCR-Official",
        "architecture_source": "untouched upstream model_ResNet.py",
        "source_commit": OFFICIAL_COMMIT,
        "seed": args.seed,
        "repeat": args.repeat,
        "checkpoint_selection": "epoch_30_final; validation_not_used_for_selection",
        "released_training_recipe": True,
        "external_evaluation_extension": (
            "deterministic released preprocessing without HSV jitter; "
            "released fusion and AP meter at batch_size=1"
        ),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    complete_marker.touch()
    print(
        f"AHCR_OFFICIAL_STRICT_RUN_OK seed={args.seed} repeat={args.repeat} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
