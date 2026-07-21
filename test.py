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


def main():
    set_seed(42)
    device = "cuda:0"
    MODEL_PATH = "./pretrained/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"
    INSTRUCT = "展现出悲苦沙哑的声音质感,语速偏慢,情绪浓烈且带有哭腔,以标准普通话缓慢诉说,情感强烈,语调哀怨高亢,音高起伏大。"
    TEXT = "这些年代受的苦，就跟你说上十天半个月也说不完。"
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    wavs, sr = tts.generate_voice_design(
        text=TEXT,
        instruct=INSTRUCT,
        language="Chinese",
    )
    sf.write("test.wav", wavs[0], sr)


if __name__ == "__main__":
    main()
