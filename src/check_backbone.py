"""
check_backbone.py  -  Load test and 4 GB fit test for the backbone (plan sections 2, 4, Amendment 1).

Uses random tensors only: no project image, label or result is touched.

1. Load the DINOv3 ViT-S/16 weights from timm/vit_small_patch16_dinov3.lvd1689m into a local cache,
   record the Hub revision and the SHA-256 of the weight file, truncate the RoPE periods to bfloat16
   (to match the original model), and check the token layout at 224 x 224.
2. Fit test: disease classifier (backbone -> CLS token -> dropout 0.20 -> Linear(384, 3)) with the
   last 3 blocks trainable and all LayerNorms frozen; one AdamW step at batch 32.
   Memory rules in order: full precision -> autocast fp16 -> gradient accumulation 2 x 16.
OUTPUT: results/backbone_check.json
"""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

import timm      # noqa: E402
import torch     # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from torch import nn  # noqa: E402

REPO = "timm/vit_small_patch16_dinov3.lvd1689m"
MODEL = "vit_small_patch16_dinov3.lvd1689m"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.20)
        self.fc = nn.Linear(backbone.num_features, 3)

    def features(self, x):
        return self.backbone.forward_features(x)[:, 0]      # CLS token after the final norm

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))


def set_phase_b(model):
    for p in model.parameters():
        p.requires_grad = False
    for block in model.backbone.blocks[-3:]:
        for p in block.parameters():
            p.requires_grad = True
    for m in model.backbone.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def try_step(model, mode, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(mode != "fp32"))
    micro = [32] if mode != "accum_2x16" else [16, 16]
    optimizer.zero_grad()
    for size in micro:
        x = torch.randn(size, 3, 224, 224, device=device)
        y = torch.randint(0, 3, (size,), device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(mode != "fp32")):
            loss = nn.functional.cross_entropy(model(x), y) * size / 32
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    info = HfApi().model_info(REPO)
    weights = hf_hub_download(REPO, "model.safetensors", revision=info.sha)
    licence = hf_hub_download(REPO, "LICENSE.md", revision=info.sha)
    backbone = timm.create_model(MODEL, pretrained=True, num_classes=0,
                                 pretrained_cfg_overlay=dict(file=weights))
    rope = getattr(backbone, "rope", None)
    if rope is not None and hasattr(rope, "periods"):
        rope.periods = rope.periods.to(torch.bfloat16).to(torch.float32)
    device = torch.device("cuda")
    model = Classifier(backbone).to(device)

    model.eval()
    with torch.inference_mode():
        tokens = model.backbone.forward_features(torch.randn(2, 3, 224, 224, device=device))
    set_phase_b(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    record = {
        "repo": REPO, "revision": info.sha, "weights_file": "model.safetensors", "weights_sha256": sha256(weights),
        "licence_sha256": sha256(licence), "timm_version": timm.__version__, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "rope_periods_truncated_to_bf16": bool(rope is not None),
        "num_blocks": len(model.backbone.blocks), "feature_dim": model.backbone.num_features,
        "token_shape_at_224": list(tokens.shape), "num_prefix_tokens": getattr(model.backbone, "num_prefix_tokens", None),
        "params_total": total, "params_trainable_phase_b": trainable,
        "params_trainable_phase_a": sum(p.numel() for p in model.fc.parameters()),
    }
    for mode in ("fp32", "amp_fp16", "accum_2x16"):
        try:
            record["fit_test_mode"], record["peak_gpu_memory_gib"] = mode, round(try_step(model, mode, device), 3)
            record["fit_test_passed"] = True
            break
        except torch.OutOfMemoryError:
            record.setdefault("fit_test_failed_modes", []).append(mode)
            record["fit_test_passed"] = False
    (ROOT / "results" / "backbone_check.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
