#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_labels(annotation_path, num_classes):
    with annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    labels = defaultdict(lambda: [0] * num_classes)
    image_names = {item["id"]: Path(item["file_name"]).name for item in data["images"]}
    category_ids = sorted(item["id"] for item in data["categories"])
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    for annotation in data["annotations"]:
        labels[image_names[annotation["image_id"]]][category_to_index[annotation["category_id"]]] = 1
    categories = [item["name"] for item in sorted(data["categories"], key=lambda item: item["id"])]
    return image_names.values(), labels, categories


def write_list(path, names, labels, root, split):
    with path.open("w", encoding="utf-8") as handle:
        for name in names:
            a_path = (root / f"{split}_A" / name).resolve()
            b_path = (root / f"{split}_B" / name).resolve()
            if not a_path.is_file() or not b_path.is_file():
                raise FileNotFoundError(f"Missing pair for {name}")
            target = ",".join(map(str, labels[name]))
            handle.write(f"{a_path} {b_path} {target}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260901)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_names, train_labels, categories = load_labels(args.dataset_root / "train.json", 12)
    test_names, test_labels, test_categories = load_labels(args.dataset_root / "test.json", 12)
    if categories != test_categories:
        raise ValueError("Train/test category definitions differ")

    names = sorted(train_names)
    random.Random(args.split_seed).shuffle(names)
    val_size = round(len(names) * args.val_ratio)
    val_names = sorted(names[:val_size])
    train_names = sorted(names[val_size:])
    test_names = sorted(test_names)

    write_list(args.output_dir / "LDXray_train.txt", train_names, train_labels, args.dataset_root, "train")
    write_list(args.output_dir / "LDXray_val.txt", val_names, train_labels, args.dataset_root, "train")
    write_list(args.output_dir / "LDXray_test.txt", test_names, test_labels, args.dataset_root, "test")
    (args.output_dir / "ldxray_classes.txt").write_text("\n".join(categories) + "\n", encoding="utf-8")
    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "test_samples": len(test_names),
        "classes": categories,
    }
    (args.output_dir / "LDXray_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
