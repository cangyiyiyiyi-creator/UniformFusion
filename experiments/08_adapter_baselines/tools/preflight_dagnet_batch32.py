#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main_finetune import build_model, get_args_parser, set_seed

out = Path("论文补充验证_20260907/02B_DAGNet_Batch32复核/preflight_batch32.json")
out.parent.mkdir(parents=True, exist_ok=True)
set_seed(20260907, deterministic=True)
args = get_args_parser().parse_args([
    "--model", "dagnet_official_adapter", "--input_size", "256",
    "--num_classes", "15", "--teacher_mode", "false",
])
model = build_model(args).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
torch.cuda.reset_peak_memory_stats()
view_a = torch.randn(32, 3, 256, 256, device="cuda")
view_b = torch.randn_like(view_a)
target = torch.rand(32, 15, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    logits = model(view_a, view_b)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
loss.backward()
optimizer.step()
torch.cuda.synchronize()
payload = {
    "status": "passed",
    "batch_size": 32,
    "input_size": 256,
    "optimizer": "AdamW",
    "forward_backward_optimizer_step": True,
    "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2**20,
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print("DAGNET_BATCH32_PREFLIGHT_OK", payload["peak_memory_MiB"])
