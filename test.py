"""Evaluate contrastive activation steering on the InstructTTS-Eval benchmark."""

from __future__ import annotations
import json
import random
import steer
import torch
from pathlib import Path
from typing import Any
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

STEERING_LAYERS = list(range(16, 24))

def _write_jsonl(output_dir: Path, split: str, entries: list[dict[str, Any]]) -> None:
    path = output_dir / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  wrote {len(entries)} entries -> {path}")


def run_steer(
    tts: Qwen3TTSModel,
    scale: float,
    gen_scale: float,
    hook_mode: str,
    num_samples: int,
    split: str,
    instr_types: list[str] = ["APS", "DSD", "RP"]
) -> None:
    OUTPUT_DIR = Path(".") / "output_steer" / split
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading dataset ...")
    ds = load_dataset("CaasiHUANG/InstructTTSEval", split=split)
    ds = ds.select_columns(["id", "text"] + instr_types).select(range(num_samples))

    entries: list[dict[str, Any]] = []
    for row in ds:
        sid: str = row["id"]
        text: str = row["text"]
        entry: dict[str, Any] = {"id": sid, "text": text}

        for it in instr_types:
            instr: str = row[it]
            print(f"[{sid}] {it}: {instr[:60]}...")

            steerings = steer.calibrate(
                tts, instr, STEERING_LAYERS,
                hook_mode=hook_mode, lang=split,
            )
            gen_rel = f"{sid}/{sid}_{it}.wav"
            wav, sr = steer.generate(
                tts,
                text,
                instr,
                steerings,
                STEERING_LAYERS,
                scale,
                hook_mode=hook_mode,
                steer_generate_scale=gen_scale,
            )
            sf.write(OUTPUT_DIR / gen_rel, wav, sr)
            print(f"       -> {gen_rel} saved")
            entry[it] = {"instruction": instr, "gen_path": gen_rel}

        entries.append(entry)

    _write_jsonl(OUTPUT_DIR, split, entries)


def run_baseline(
    tts: Qwen3TTSModel,
    num_samples: int,
    split: str,
    instr_types: list[str] | None = None,
) -> None:
    OUTPUT_DIR = Path(".") / "output_baseline" / split
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading dataset ...")
    ds = load_dataset("CaasiHUANG/InstructTTSEval", split=split)
    ds = ds.select_columns(["id", "text"] + instr_types).select(range(num_samples))

    entries: list[dict[str, Any]] = []
    for row in ds:
        sid: str = row["id"]
        text: str = row["text"]
        entry: dict[str, Any] = {"id": sid, "text": text}

        for it in instr_types:
            instr: str = row[it]
            print(f"[{sid}] {it}: {instr[:60]}...")

            wav, sr = tts.generate_voice_design(
                text=text, instruct=instr, language="auto"
            )
            gen_rel = f"{sid}/{sid}_{it}.wav"
            sf.write(OUTPUT_DIR / gen_rel, wav[0], sr)
            print(f"       -> {gen_rel} saved")
            entry[it] = {"instruction": instr, "gen_path": gen_rel}

        entries.append(entry)

    _write_jsonl(OUTPUT_DIR, split, entries)


def run_gt(
    num_samples: int,
    split: str,
    instr_types: list[str] | None = None,
) -> None:
    OUTPUT_DIR = Path(".") / "output_gt" / split
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading dataset ...")
    ds = load_dataset("CaasiHUANG/InstructTTSEval", split=split)
    ds = ds.select(range(num_samples))

    entries: list[dict[str, Any]] = []
    for row in ds:
        sid: str = row["id"]
        text: str = row["text"]
        audio = row["reference_audio"]
        gen_rel = f"{sid}/{sid}.wav"
        sf.write(OUTPUT_DIR / gen_rel, audio["array"], audio["sampling_rate"])
        print(f"[{sid}] gt saved -> {gen_rel}")

        entry: dict[str, Any] = {"id": sid, "text": text}
        for it in instr_types:
            entry[it] = {"instruction": row[it], "gen_path": gen_rel}
        entries.append(entry)

    _write_jsonl(OUTPUT_DIR, split, entries)


if __name__ == "__main__":
    set_seed(42)
    MODEL_PATH = "./pretrained/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    run_steer(tts, scale=1.0, gen_scale=0.5, split="zh", hook_mode="pa", num_samples=1)
    run_baseline(tts, num_samples=1, split="zh")
    run_gt(num_samples=1, split="zh")
