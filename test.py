"""Evaluate contrastive activation steering on the InstructTTS-Eval benchmark."""

from __future__ import annotations
import os
import steer
import random
import torch
from pathlib import Path
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from datasets import load_dataset

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for deterministic output."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() -> None:
    device = "cuda:0"
    MODEL_PATH = Path(".") / "pretrained" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    # Steering configuration
    STEERING_LAYERS = list(range(16, 24))
    SCALE: float = 1.0
    GEN_SCALE: float = 0.5
    HOOK_MODE = "pa"

    # Evaluation subset
    INSTRUCTION_TYPE: str = "DSD"  # "APS" | "DSD" | "RP"
    NUM_SAMPLES: int = 1
    SPLIT: str = "zh"
    OUTPUT_DIR = Path(".") / "output" / f"{SPLIT.lower()}"
    OUTPUT_DIR.mkdir(parents=True ,exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(
        os.fspath(MODEL_PATH),
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("loading dataset ...")
    ds_all = load_dataset("CaasiHUANG/InstructTTSEval", split=SPLIT)
    ds = ds_all.select_columns(["id", "text", INSTRUCTION_TYPE]).select(
        range(NUM_SAMPLES)
    )
    for row in ds:
        sid: str = row["id"]
        text: str = row["text"]
        instr: str = row[INSTRUCTION_TYPE]
        print(f"[{sid}] text: {text}")
        print(f"       {INSTRUCTION_TYPE}:  {instr}")
        sample_dir = OUTPUT_DIR / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Per-sample calibration (lang auto-selects 10 calibration texts)
        steerings = steer.calibrate(
            tts,
            instr,
            STEERING_LAYERS,
            hook_mode=HOOK_MODE,
            lang=SPLIT,
        )
        norms = [f"{steerings[l].norm().item():.1f}" for l in STEERING_LAYERS]
        print(f"       [{HOOK_MODE}] steering norms: {norms}")

        tag = (f"{INSTRUCTION_TYPE}_{HOOK_MODE}_s{SCALE:.1f}_gens{GEN_SCALE:.1f}")
        wav, sr = steer.generate(
            tts,
            text,
            instr,
            steerings,
            STEERING_LAYERS,
            SCALE,
            hook_mode=HOOK_MODE,
            steer_generate_scale=GEN_SCALE,
        )
        fname = sample_dir / f"{sid}_{tag}.wav"
        sf.write(fname, wav, sr)
        print(f"       -> {fname} saved")

if __name__ == "__main__":
    main()
