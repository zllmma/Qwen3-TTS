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
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from datasets import load_dataset
import os
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_steering_vector(tts, instruct, target_layers, hook_mode="pa"):
    hidden_outputs = {l: {} for l in target_layers}
    handles = []

    if hook_mode == "pa":
        def make_hook(l):
            def hook(module, input, output):
                if input[0].shape[1] > 1:
                    hidden_outputs[l]["val"] = input[0].detach().clone()
            return hook
        for l in target_layers:
            h = tts.model.talker.model.layers[l].post_attention_layernorm.register_forward_hook(make_hook(l))
            handles.append(h)
    else:
        def make_hook(l):
            def hook(module, input, output):
                if input[0].shape[1] > 1:
                    hidden_outputs[l]["val"] = output[0].detach().clone()
            return hook
        for l in target_layers:
            h = tts.model.talker.model.layers[l].register_forward_hook(make_hook(l))
            handles.append(h)

    try:
        tts.generate_voice_design(
            text="test",
            instruct=instruct,
            language="Chinese",
            max_new_tokens=3,
        )
        return {l: hidden_outputs[l]["val"][0, -1, :] for l in target_layers}
    finally:
        for h in handles:
            h.remove()


def get_token_count(tts, instruct):
    if instruct is None or instruct == "":
        return 0
    return tts._tokenize_texts([tts._build_instruct_text(instruct)])[0].shape[-1]


def generate_with_steering(tts, text, instruct, steerings, steering_layers, scale, n_inst, hook_mode="pa"):
    handles = []

    if hook_mode == "pa":
        def make_hook(l, s, n):
            sv = steerings[l] / (steerings[l].norm() + 1e-8)
            def pre_hook(module, input):
                if input[0].shape[1] > 1:
                    input[0][:, :n, :] += s * sv.to(
                        device=input[0].device, dtype=input[0].dtype
                    )
            return pre_hook
        for l in steering_layers:
            h = tts.model.talker.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
                make_hook(l, scale, n_inst)
            )
            handles.append(h)
    else:
        def make_hook(l, s, n):
            sv = steerings[l] / (steerings[l].norm() + 1e-8)
            def hook(module, input, output):
                if input[0].shape[1] > 1:
                    output[0][:, :n, :] += s * sv.to(
                        device=output[0].device, dtype=output[0].dtype
                    )
            return hook
        for l in steering_layers:
            h = tts.model.talker.model.layers[l].register_forward_hook(
                make_hook(l, scale, n_inst)
            )
            handles.append(h)

    try:
        wavs, sr = tts.generate_voice_design(
            text=text,
            instruct=instruct,
            language="Chinese",
        )
        return wavs[0], sr
    finally:
        for h in handles:
            h.remove()


def calibrate_steering(tts, instruct, target_layers, hook_mode="pa"):
    h_strong = extract_steering_vector(tts, instruct, target_layers, hook_mode=hook_mode)
    h_neutral = extract_steering_vector(tts, "", target_layers, hook_mode=hook_mode)
    return {l: h_strong[l] - h_neutral[l] for l in target_layers}


def main():
    device = "cuda:0"
    MODEL_PATH = "./pretrained/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"
    STEERING_LAYERS = list(range(16, 24))
    SCALE = 1.0
    NUM_SAMPLES = 10

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    set_seed(42)

    print("loading dataset ...")
    ds = load_dataset("CaasiHUANG/InstructTTSEval", split="zh", streaming=False)
    ds = ds.select_columns(["id", "text", "DSD"]).select(range(NUM_SAMPLES))

    os.makedirs("./instructtts_test_output", exist_ok=True)

    for row in ds:
        sid = row["id"]
        text = row["text"]
        dsd = row["DSD"]
        n_inst = get_token_count(tts, dsd)

        print(f"[{sid}] text: {text[:60]}...")
        print(f"       DSD:  {dsd[:60]}...")

        for hook_mode, tag_suffix in [("ffn", "ffn"), ("pa", "pa")]:
            steerings = calibrate_steering(tts, dsd, STEERING_LAYERS, hook_mode=hook_mode)
            norms = [f"{steerings[l].norm().item():.1f}" for l in STEERING_LAYERS]
            print(f"       [{hook_mode}] steering norms: {norms}")

            for layers, scale in [
                ([], 0.0),
                (STEERING_LAYERS, SCALE),
            ]:
                tag = "baseline" if layers == [] else f"steer_l16-24_s{SCALE:.1f}_{tag_suffix}"
                set_seed(42)
                wav, sr = generate_with_steering(tts, text, dsd, steerings, layers, scale, n_inst, hook_mode=hook_mode)
                fname = f"./instructtts_test_output/{sid}_{tag}.wav"
                sf.write(fname, wav, sr)
                print(f"       -> {sid}_{tag}.wav saved")
        print()


if __name__ == "__main__":
    main()
