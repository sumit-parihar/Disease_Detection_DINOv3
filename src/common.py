"""
common.py  -  Shared paths, the DINOv3 disease classifier and the link to the frozen original code.

The original project is only READ:
  * its src/ folder is put on sys.path so dataset.py (same transforms, same split file) and config.py
    (same hyper-parameters) are used unchanged; sys.dont_write_bytecode stops Python from writing
    __pycache__ into it;
  * the split file is hash-checked before any image is opened.
All outputs go to this folder (models/ is ignored by Git, results/ is kept).
"""

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # weights are already cached; never re-download

FROZEN_ROOT = Path("D:/Capstone Project/Capstone Project")
EXTENDED_ROOT = Path("D:/capstone-paper-extended")
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SPLIT_SHA256 = "d59a80cce32c84ed32d9fc6ef5c5d870082933cc8892c757184d16d4052a6401"
SEEDS = (42, 123, 2026)

# backbone (results/backbone_check.json)
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
HUB_REPO = "timm/vit_small_patch16_dinov3.lvd1689m"
HUB_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
WEIGHTS_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
PHASE_B_BLOCKS = 3


def use_frozen_code():
    """Make the original src/ importable without writing anything into it."""
    sys.dont_write_bytecode = True
    path = str(FROZEN_ROOT / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_split():
    found = sha256(FROZEN_ROOT / "splits" / "split_assignments.csv")
    if found != SPLIT_SHA256:
        sys.exit(f"Split file hash mismatch: {found}")
    return found


import torch           # noqa: E402
from torch import nn   # noqa: E402


def load_backbone():
    """DINOv3 ViT-S/16 from the local cache at the pinned revision, weight hash checked,
    RoPE periods truncated to bfloat16 (plan, Amendment 1)."""
    import timm
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(HUB_REPO, "model.safetensors", revision=HUB_REVISION)
    if sha256(weights) != WEIGHTS_SHA256:
        sys.exit("DINOv3 weight file hash mismatch")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0,
                                 pretrained_cfg_overlay=dict(file=weights))
    backbone.rope.periods = backbone.rope.periods.to(torch.bfloat16).to(torch.float32)
    assert not backbone.rope.aug_active, "RoPE training augmentation must be off"
    return backbone


class DinoClassifier(nn.Module):
    """backbone -> CLS token after the final norm (384) -> dropout 0.20 -> Linear(384, 3).
    Same interface as the original DiseaseClassifier (extract_features, set_phase_a, set_phase_b)."""

    def __init__(self, backbone=None, dropout=0.20, num_classes=3):
        super().__init__()
        self.backbone = backbone if backbone is not None else load_backbone()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def extract_features(self, images):
        return self.backbone.forward_features(images)[:, 0]

    def forward(self, images):
        return self.fc(self.dropout(self.extract_features(images)))

    def set_phase_a(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = True

    def set_phase_b(self):
        """Last 3 transformer blocks + output layer learn; every LayerNorm stays frozen."""
        self.set_phase_a()
        for block in self.backbone.blocks[-PHASE_B_BLOCKS:]:
            for p in block.parameters():
                p.requires_grad = True
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_classifier(path, device):
    """Rebuild a trained classifier from a checkpoint (architecture from the hashed weights)."""
    model = DinoClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    return model.to(device).eval()
