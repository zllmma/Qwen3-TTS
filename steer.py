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
"""Contrastive activation steering for Qwen3-TTS Voice Design.

Usage::

    import steer
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(...)
    steer.set_seed(42)

    # Calibrate: extract per-layer steering vectors from an instruction.
    layers = list(range(16, 24))
    steerings = steer.calibrate(tts, instruct, layers, hook_mode="pa")

    # Generate with steering.
    wav, sr = steer.generate(tts, text, instruct, steerings, layers, scale=1.0)
"""

from __future__ import annotations

import torch
from typing import Dict, List


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for deterministic output."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_token_count(tts, instruct: str) -> int:
    """Return the number of tokens in ``instruct`` after chat-template wrapping."""
    if instruct is None or instruct == "":
        return 0
    return tts._tokenize_texts([tts._build_instruct_text(instruct)])[0].shape[-1]


def extract_hidden(
    tts,
    instruct: str,
    target_layers: List[int],
    hook_mode: str = "pa",
    calibration_text: str = "test",
) -> Dict[int, torch.Tensor]:
    """Run a short prefill and capture hidden states at *target_layers*.

    ``hook_mode``:
        ``"pa"`` — hook ``post_attention_layernorm``, read pre-norm input
            (post-attention + residual, before FFN).
        ``"ffn"`` — hook the full ``DecoderLayer``, read post-FFN output.

    ``calibration_text``:
        Short text used to build the prefill sequence.  Using different texts
        and averaging the resulting steering vectors reduces calibration bias.
    """
    hidden_outputs: Dict[int, dict] = {l: {} for l in target_layers}
    handles: list = []

    if hook_mode == "pa":
        def _hook_factory(l: int):
            def hook(module, input, output):
                if input[0].shape[1] > 1:          # prefill only
                    hidden_outputs[l]["val"] = input[0].detach().clone()
            return hook
        for l in target_layers:
            h = tts.model.talker.model.layers[l].post_attention_layernorm.register_forward_hook(
                _hook_factory(l)
            )
            handles.append(h)
    else:
        def _hook_factory(l: int):
            def hook(module, input, output):
                if input[0].shape[1] > 1:
                    hidden_outputs[l]["val"] = output[0].detach().clone()
            return hook
        for l in target_layers:
            h = tts.model.talker.model.layers[l].register_forward_hook(_hook_factory(l))
            handles.append(h)

    try:
        tts.generate_voice_design(
            text=calibration_text,
            instruct=instruct,
            language="auto",
        )
        # Return the hidden state at the *last* (codec-bos) position on every layer.
        return {l: hidden_outputs[l]["val"][0, -1, :] for l in target_layers}
    finally:
        for h in handles:
            h.remove()


CALIBRATION_TEXTS = {
    "zh": [
        "开始。", "你好。", "请朗读。", "这是一段测试。", "今天天气不错。",
        "现在开始。", "请继续。", "测试一下。", "读出来。", "说一句话。",
    ],
    "en": [
        "hello.", "please read.", "a test.", "good morning.",
        "one two three.", "how are you.", "thank you.", "nice day.",
        "ok.", "start.",
    ],
}


def calibrate(
    tts,
    instruct: str,
    target_layers: List[int],
    hook_mode: str = "pa",
    calibration_texts: List[str] | None = None,
    lang: str | None = None,
) -> Dict[int, torch.Tensor]:
    """Compute per-layer steering vectors for *instruct*.

    steering[l] = h_strong[l] - h_neutral[l]

    *calibration_texts* — short texts used for the calibration prefill.
    Multiple texts are averaged to reduce single-text bias.  When ``None``,
    the default set for *lang* is used (``CALIBRATION_TEXTS[lang]``).
    *lang* has no effect when *calibration_texts* is given explicitly.
    """
    if calibration_texts is None:
        if lang is None:
            lang = "zh"
        calibration_texts = CALIBRATION_TEXTS.get(lang, CALIBRATION_TEXTS["zh"])

    all_h_strong: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
    all_h_neutral: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}

    for ct in calibration_texts:
        h_strong = extract_hidden(tts, instruct, target_layers, hook_mode=hook_mode, calibration_text=ct)
        h_neutral = extract_hidden(tts, "", target_layers, hook_mode=hook_mode, calibration_text=ct)
        for l in target_layers:
            all_h_strong[l].append(h_strong[l])
            all_h_neutral[l].append(h_neutral[l])

    return {
        l: (
            torch.stack(all_h_strong[l]).mean(dim=0)
            - torch.stack(all_h_neutral[l]).mean(dim=0)
        )
        for l in target_layers
    }


def generate(
    tts,
    text: str,
    instruct: str,
    steerings: Dict[int, torch.Tensor],
    steering_layers: List[int],
    scale: float,
    hook_mode: str = "pa",
):
    """Generate speech with contrastive activation steering.

    During prefill, each layer *l* in *steering_layers* has its hidden state
    at instruction token positions shifted by ``scale * steerings[l]``
    (normalised to unit length).  The shifted representations are written
    into the KV-cache and influence every subsequent auto-regressive step.

    ``hook_mode`` selects where the shift is applied:
        ``"pa"`` — add before ``post_attention_layernorm`` (post-attn).
        ``"ffn"`` — add to the full ``DecoderLayer`` output (post-FFN).

    Returns
    -------
    (waveform: np.ndarray, sample_rate: int)
    """
    handles: list = []
    n_inst = _get_token_count(tts, instruct)

    if hook_mode == "pa":
        # Pre-hook on post_attention_layernorm: modify input *before* the norm.
        def _hook_factory(l: int, s: float, n: int):
            sv = steerings[l] / (steerings[l].norm() + 1e-8)  # unit-vector
            def pre_hook(module, input):
                if input[0].shape[1] > 1:
                    input[0][:, :n, :] += s * sv.to(
                        device=input[0].device, dtype=input[0].dtype
                    )
            return pre_hook
        for l in steering_layers:
            h = tts.model.talker.model.layers[l].post_attention_layernorm.register_forward_pre_hook(
                _hook_factory(l, scale, n_inst)
            )
            handles.append(h)
    else:
        # Post-hook on DecoderLayer: modify output in-place after the layer completes.
        def _hook_factory(l: int, s: float, n: int):
            sv = steerings[l] / (steerings[l].norm() + 1e-8)
            def hook(module, input, output):
                if input[0].shape[1] > 1:
                    output[0][:, :n, :] += s * sv.to(
                        device=output[0].device, dtype=output[0].dtype
                    )
            return hook
        for l in steering_layers:
            h = tts.model.talker.model.layers[l].register_forward_hook(
                _hook_factory(l, scale, n_inst)
            )
            handles.append(h)

    try:
        wavs, sr = tts.generate_voice_design(
            text=text,
            instruct=instruct,
            language="auto",
        )
        return wavs[0], sr
    finally:
        for h in handles:
            h.remove()


def load_universal(path: str = "./universal_steering.pkl") -> dict[int, torch.Tensor]:
    """Load a pre-computed universal steering vector from disk.

    Usage::

        universal = steer.load_universal()
        wav, sr = steer.generate(tts, text, instruct, universal, layers, scale=1.0)
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {int(k): v for k, v in data["universal_steering"].items()}
