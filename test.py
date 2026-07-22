# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate contrastive activation steering on the InstructTTS-Eval benchmark."""

from __future__ import annotations

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from datasets import load_dataset
import os
import steer


def main() -> None:
    device = "cuda:0"
    MODEL_PATH = "./pretrained/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"

    # Steering configuration
    STEERING_LAYERS: list[int] = list(range(16, 24))   # layers 16–23
    SCALE: float = 1.0                                  # steering strength

    # Evaluation subset
    INSTRUCTION_TYPE: str = "APS"            # "APS" | "DSD" | "RP"
    NUM_SAMPLES: int = 5
    OUTPUT_DIR: str = f"./instructtts_test_output/{INSTRUCTION_TYPE.lower()}"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print("loading dataset ...")
    ds = load_dataset("CaasiHUANG/InstructTTSEval", split="zh", streaming=False)
    ds = ds.select_columns(["id", "text", INSTRUCTION_TYPE]).select(range(NUM_SAMPLES))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for row in ds:
        sid: str = row["id"]
        text: str = row["text"]
        instr: str = row[INSTRUCTION_TYPE]

        print(f"[{sid}] text: {text[:60]}...")
        print(f"       {INSTRUCTION_TYPE}:  {instr[:60]}...")

        # Try both hook points: post-FFN ("ffn") and post-attention ("pa")
        for hook_mode, tag_suffix in [("ffn", "ffn"), ("pa", "pa")]:
            # Per-sample calibration
            steerings = steer.calibrate(tts, instr, STEERING_LAYERS, hook_mode=hook_mode)
            norms = [f"{steerings[l].norm().item():.1f}" for l in STEERING_LAYERS]
            print(f"       [{hook_mode}] steering norms: {norms}")

            # Baseline and steered generation
            for layers, scale in [
                ([], 0.0),                           # no steering (baseline)
                (STEERING_LAYERS, SCALE),            # steering on
            ]:
                tag = (
                    "baseline" if layers == []
                    else f"steer_l16-24_s{SCALE:.1f}_{tag_suffix}"
                )
                steer.set_seed(42)
                wav, sr = steer.generate(
                    tts, text, instr, steerings, layers, scale, hook_mode=hook_mode,
                )
                fname = f"{OUTPUT_DIR}/{sid}_{tag}.wav"
                sf.write(fname, wav, sr)
                print(f"       -> {sid}_{tag}.wav saved")
        print()


if __name__ == "__main__":
    main()
