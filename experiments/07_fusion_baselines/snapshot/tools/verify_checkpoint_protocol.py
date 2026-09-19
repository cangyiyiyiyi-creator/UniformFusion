import argparse
import hashlib
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--expected-val-list", required=True)
    args = parser.parse_args()

    expected = Path(args.expected_val_list).resolve()
    if not expected.is_file():
        raise FileNotFoundError(f"missing expected validation list: {expected}")
    expected_sha256 = file_sha256(expected)

    for checkpoint_name in args.checkpoint:
        checkpoint_path = Path(checkpoint_name)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored_args = payload.get("args", {})
        if not isinstance(stored_args, dict):
            stored_args = vars(stored_args)
        stored_name = stored_args.get("val_list")
        if not stored_name:
            raise ValueError(f"checkpoint has no val_list: {checkpoint_path}")
        stored = Path(stored_name).resolve()
        if stored != expected:
            raise ValueError(
                f"checkpoint selection split mismatch: {checkpoint_path}; "
                f"stored={stored}, expected={expected}"
            )
        if file_sha256(stored) != expected_sha256:
            raise ValueError(f"validation split checksum mismatch: {stored}")
        print(
            "CHECKPOINT_PROTOCOL_OK "
            f"checkpoint={checkpoint_path} val_sha256={expected_sha256}"
        )


if __name__ == "__main__":
    main()
