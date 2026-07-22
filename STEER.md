# Contrastive Activation Steering for Qwen3-TTS Voice Design

## 概述

本方案通过 **对比激活引导 (Contrastive Activation Steering)** 增强 Voice Design 模型对声音描述指令的遵循度。

核心思想：用模型自身在"有指令"和"无指令"两种状态下的内部表示差异，提取出指令语义的方向向量 (steering vector)，然后在推理时将模型沿该方向推动，放大指令的影响。

整个过程**不需要额外训练**，通过 PyTorch hook 机制在线完成校准和注入。

---

## 动机：为什么需要 Steering

Voice Design 模型是 decoder-only 自回归语言模型，指令文本通过 self-attention 影响后续 codec token 的生成。但实际使用中发现，模型对指令的遵循度不够强——指令写了"悲苦沙哑"，生成的语音可能只有轻微差异，甚至被后续 token 预测过程"稀释"。

问题的根源在于：指令信号在 24 层 transformer 中从浅到深逐步衰减或与文本信号混合。Steering 的思路是在指令信号最强的中间层，主动把它"推一把"。

---

## 原理：三步走

### 第一步：校准 — 提取指令方向

对每一条指令，跑两次极短的 prefill（`max_new_tokens=3`，只为了触发 prefill，不关心输出）：

```
含指令:  [instruction tokens] [assistant] [control] [简短 text] [codec_bos]
空指令:                        [assistant] [control] [简短 text] [codec_bos]
```

在目标层的 prefill 输出中，取 **codec_bos 位置**的 hidden state。codec_bos 是自回归生成开始前最后一个 token，通过 causal attention 看到了前面所有 token（包括 instruction）。两个 hidden state 的差：

```
steering[layer] = h_strong[layer] − h_neutral[layer]
```

就是"这段指令让模型在 layer 层准备生成什么不一样的东西"——指令语义在这层的编码方向。

校准在每个样本上独立进行：不同指令提取出不同的 steering vector，保证方向与指令内容对齐。

### 第二步：注入 — 推动指令 token 的中间表示

推理时，在 **prefill 阶段**对目标层的 instruction token 位置加注：

```
hidden[layer][instruction_tokens] += scale × steering[layer] / ‖steering[layer]‖
```

- 每层 steering 归一化为单位向量，用 `scale` 控制幅度
- 注入发生在 prefill——修改后的 hidden state 被写入 KV cache
- 后续每一步 auto-regressive generate，新 token 通过 self-attention 看到的就是"被增强过的"指令

### 第三步：生成 — 标准 decode

注入完成后，推理走标准流程：Talker 自回归生成第一个码本，Code Predictor 补全剩余码本，Tokenizer decoder 解码为波形。整个过程除了 prefill 阶段的一次性注入，没有额外计算开销。

---

## 架构细节

### 干预点：post-attention vs post-FFN

一个 DecoderLayer 的内部结构：

```
Input
  → input_layernorm → self_attn → + residual    ──→ post_attention_layernorm → FFN → + residual → Output
                         ↑ post-attention (pa)                                           ↑ post-FFN (ffn)
```

提供两种 hook 模式：

| 模式 | Hook 位置 | 提取方式 | 注入方式 |
|------|----------|---------|---------|
| `"pa"` | `post_attention_layernorm` 输入 | forward_hook 读 `input[0]` | forward_pre_hook 改 `input[0]` |
| `"ffn"` | DecoderLayer 输出 | forward_hook 读 `output[0]` | forward_hook 改 `output[0]` |

`"pa"` 更纯粹——只捕捉 attention 层面的语义信息，没被 FFN 的 token 预测加工搅乱。`"ffn"` 包含了 FFN 对指令的进一步加工，信号更强但噪声也可能更高。

### 干预层范围：16-23

24 层模型中，steering norm 随层深单调增长：

```
层 0-7:  norm < 8      → 指令信号太弱，干预无效
层 8-15: norm 10-27    → 语义编码中，但尚不稳定
层 16-23: norm 34-131  → 指令信号强且仍在语义空间 → 最优干预区间
```

选择 16-23 是因为：
- 太浅（<16）：指令语义还没充分编码
- 太深（>23）：hidden 已经高度特化为 token prediction，修改容易破坏生成质量
- 16-23：语义充沛但尚未锁死，有足够的修改余地

每层提取各自的 steering vector 并归一化，所有修改通过 KV cache 向下游逐层传播。

---

## 与其它方法的区别

| 方法 | 对比对象 | 为什么无效/不够好 |
|------|---------|------------------|
| CFG (Classifier-Free Guidance) | — | 需要训练时指令 dropout，当前模型无此能力 |
| DoLa (层间 logits 对比) | 同一 forward 内深层 vs 浅层 | `codec_head` 只在最后一层训练，浅层 logits 是噪声 |
| Hidden 层间对比 | 同一 forward 内深层 hidden vs 浅层 hidden | 层间差异是 token prediction refinement，不是指令语义 |
| Instruction Embedding Amplification | 同一指令内 token 偏离均值 | 放大偏离是把独特词推向随机噪声方向 |
| **Steering Vector** | **不同输入之间**的 hidden state | **语义层面的真实差异**，方向有明确含义 |

关键区别：本方案对比的是"有指令"和"无指令"两个**独立的 forward pass**，捕捉的是条件差异而非同一次 forward 的内部加工差异。

---

## API

```python
import steer

# 校准：为一段指令提取 steering vectors
layers = list(range(16, 24))
steerings = steer.calibrate(
    tts,
    instruct="低沉磁性的男声，语速偏慢。",
    target_layers=layers,
    hook_mode="pa",   # "pa" (post-attn) 或 "ffn" (post-FFN)
)
# steerings = {16: tensor(2048,), 17: tensor(2048,), ..., 23: tensor(2048,)}

# 生成：注入 steering 并合成语音
wav, sr = steer.generate(
    tts,
    text="你好，欢迎使用语音合成。",
    instruct="低沉磁性的男声，语速偏慢。",
    steerings=steerings,
    steering_layers=layers,
    scale=1.0,
    hook_mode="pa",
)
# wav: np.ndarray, sr: int

# 固定随机种子以保证可复现
steer.set_seed(42)
```

---

## 局限与改进方向

1. **每样本需要两次校准 forward**（强指令 + 空指令），增加了 2× 短 prefill 的开销。对于大批量评测，校准开销显著
2. **校准文本固定为 "test"**，长文本生成时校准和推理的文本长度不匹配，可能影响 steering 精度
3. **单指令校准**——steering 只对当前指令有效。跨指令泛化需要多指令平均或元学习
4. **FFN vs PA 的最优选择**尚未量化，当前依赖主观听感对比
5. **scale 超参数**需要调优，不同指令类型（APS/DSD/RP）的最优 scale 可能不同
