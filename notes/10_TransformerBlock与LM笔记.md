# TransformerBlock 和 TransformerLM 从零读懂

> 基于 `nn_utils.py` 中的两个类，逐行讲解把前面所有模块**组装成完整 Transformer** 的最后两步。

---

## 一、整体脉络

前面 9 篇笔记分别讲了各个模块，现在是把它们**拼起来**的时候了：

```
TransformerBlock:  一层 Transformer（Attention + FFN + 残差 + Pre-norm）
        ↓
TransformerLM:    多层 TransformerBlock 堆叠 + Embedding + LM Head
        ↓
      完整的语言模型
```

| 前续笔记 | 在这个类里的角色 |
|:---------|:---------------|
| `08_多头自注意力笔记.md` + `07_RoPE` | `MultiHeadSelfAttention`（带 RoPE） |
| `09_前馈网络笔记.md` | `SwiGLU` FFN |
| `04_RMSNorm层笔记.md` | `RMSNorm` Pre-norm |
| `02_Linear层笔记.md` | LM Head 输出投影 |
| `03_Embedding层笔记.md` | Token Embedding |

---

## 二、TransformerBlock——一个 Transformer 层

### 架构

```
输入 x
    │
    ├── RMSNorm_1(x)
    ├── MultiHeadSelfAttention(ln1(x))   ← 带 RoPE
    ├── + 残差连接 (x + attn_out)
    │
    ├── RMSNorm_2(x)
    ├── SwiGLU(ln2(x))
    ├── + 残差连接 (x + ffn_out)
    │
    输出 (形状不变)
```

这种结构叫 **Pre-norm**——归一化放在子层**前面**，而不是后面。

> Post-norm（原始 Transformer）：`x = LN(x + Attention(x))`
> Pre-norm（现代 LLM）：`x = x + Attention(LN(x))`
>
> Pre-norm 的梯度更稳定，训练更简单。

### `__init__`：三个子模块

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, ...):
        super().__init__()
        self.ln1 = RMSNorm(d_model)                  # 第一个 Pre-norm
        self.ln2 = RMSNorm(d_model)                  # 第二个 Pre-norm
        self.attn = MultiHeadSelfAttention(          # 多头自注意力（带 RoPE）
            d_model, num_heads, use_rope=True,
            theta=theta, max_seq_len=max_seq_len,
        )
        self.ffn = SwiGLU(d_model, d_ff)             # 前馈网络
```

| 参数 | 含义 | 传递给 |
|:----|:-----|:-------|
| `d_model` | 模型维度 | RMSNorm、MHA、SwiGLU |
| `num_heads` | 头数 | MHA |
| `d_ff` | FFN 内部维度 | SwiGLU |
| `max_seq_len` | 最大序列长度 | MHA（给 RoPE 用） |
| `theta` | RoPE 基础频率 | MHA（给 RoPE 用） |

### `forward`：Pre-norm 残差结构

```python
def forward(self, x):
    seq_len = x.shape[-2]                         # 取序列长度
    token_position = torch.arange(seq_len)         # 生成位置索引 0,1,2,...,S-1

    # MHA 子层：Pre-norm + 残差
    a = x + self.attn(self.ln1(x), token_positions=token_position)

    # FFN 子层：Pre-norm + 残差
    b = a + self.ffn(self.ln2(a))

    return b
```

**逐行拆解**

**① 获取序列长度**

```python
seq_len = x.shape[-2]
```

`x` 形状是 `(B, S, d_model)`，`-2` 就是 `S`（序列长度）。用于生成 RoPE 需要的位置索引。

**② 生成位置索引**

```python
token_position = torch.arange(seq_len)
```

生成 `[0, 1, 2, ..., S-1]`，传给 `self.attn` 中的 RoPE。

**③ MHA 子层**

```python
a = x + self.attn(self.ln1(x), token_positions=token_position)
```

Pre-norm 结构：先归一化，再做 Attention，然后加上原始输入（残差连接）。

```
执行顺序:        x → ln1 → attn → + x → a
等价公式:  a = x + Attention(RMSNorm(x))
```

为什么 Pre-norm 好？梯度可以直接通过残差连接传到前面层，不受 Attention 内部的矩阵乘法影响。

**④ FFN 子层**

```python
b = a + self.ffn(self.ln2(a))
```

同上：先归一化，再做 FFN，然后加残差。FFN 对每个 token 独立做非线性变换。

```
执行顺序:        a → ln2 → ffn → + a → b
等价公式:  b = a + SwiGLU(RMSNorm(a))
```

---

## 三、TransformerLM——完整的语言模型

### 架构

```
输入 token_ids: (B, S)                    ← 整数 ID
    │
    ├── Embedding(vocab_size, d_model)     → (B, S, d_model)
    │
    ├── TransformerBlock × N              → (B, S, d_model)
    │      ...
    │
    ├── RMSNorm(d_model)                  → (B, S, d_model)
    │
    ├── Linear(d_model, vocab_size)        → (B, S, vocab_size) ← logits
    │
    输出 logits: (B, S, vocab_size)        ← 每个位置预测下一个词的概率
```

### `__init__`：四大组件

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model,
                 num_layers, num_heads, d_ff, rope_theta, ...):
        super().__init__()

        # ① Token Embedding
        self.token_embeddings = Embedding(vocab_size, d_model)

        # ② N 层 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff,
                             context_length, rope_theta)
            for _ in range(num_layers)
        ])

        # ③ 最终 RMSNorm
        self.ln_final = RMSNorm(d_model)

        # ④ LM Head（输出投影到词汇表大小）
        self.lm_head = Linear(d_model, vocab_size)
```

| 参数 | 含义 | 典型值 |
|:----|:-----|:-------|
| `vocab_size` | 词汇表大小 | 50257 (GPT-2) |
| `context_length` | 最大序列长度 | 2048 |
| `d_model` | 模型维度 | 768 |
| `num_layers` | Transformer 层数 | 12 |
| `num_heads` | 注意力头数 | 12 |
| `d_ff` | FFN 内部维度 | 3072 |
| `rope_theta` | RoPE 频率 | 10000 |

### `forward`：前向传播

```python
def forward(self, in_indices):
    # ① Embedding 查表
    x = self.token_embeddings(in_indices)      # (B, S) → (B, S, d_model)

    # ② 逐层过 TransformerBlock
    for i in range(len(self.layers)):
        x = self.layers[i](x)                   # 形状不变: (B, S, d_model)

    # ③ 最终 RMSNorm
    x = self.ln_final(x)                        # 形状不变: (B, S, d_model)

    # ④ LM Head → 输出 logits
    return self.lm_head(x)                      # (B, S, d_model) → (B, S, vocab_size)
```

**逐行拆解**

**① Embedding**

```python
x = self.token_embeddings(in_indices)
```

`in_indices` 形状 `(B, S)`，每个位置存一个 token ID（整数）。Embedding 层查出对应的向量，输出 `(B, S, d_model)`。

**② 逐层传递**

```python
for i in range(len(self.layers)):
    x = self.layers[i](x)
```

每层输出形状和输入相同 `(B, S, d_model)`，所以可以串联。`N` 层就是 `N` 个 TransformerBlock 依次处理。

这里用 `nn.ModuleList` 而不是普通 Python 列表，因为 `nn.ModuleList` 让 PyTorch 能正确发现所有子模块的参数。

**③ 最终 RMSNorm**

```python
x = self.ln_final(x)
```

在所有 Transformer 层之后再加一次 RMSNorm。作用是稳定最后一层的输出分布，让 LM Head 的输入更规范。

**④ LM Head**

```python
return self.lm_head(x)
```

一个 Linear 层，把 `d_model` 维映射到 `vocab_size` 维，输出每个 token 位置上所有候选词的**分数**（logits）。

后续对这个 logits 做 `softmax` 就得到每个位置上预测下一个词的概率分布。

```
logits:  (B, S, vocab_size)
               ↓ softmax(dim=-1)
probs:   (B, S, vocab_size) ← 每个位置对 vocab 中每个词的概率
               ↓ argmax
pred:    (B, S)            ← 每个位置预测的下一个 token ID
```

---

## 四、完整数据流

```
输入句子: "The cat sat"

BPE Tokenizer → token_ids [1523, 42, 7]    ← 形状 (B=1, S=3)
    │
    ▼ Embedding
    手 → 向量, 向量, 向量                   ← 形状 (1, 3, d_model)
    │
    ▼ TransformerBlock × N
    每层: Attention(RMSNorm(x)) + x
          SwiGLU(RMSNorm(x)) + x
    层 1: 吸收了上下文信息的向量               ← 形状不变
    层 2: 更深层的表示                       ← 形状不变
    ...
    层 N: 表示                              ← 形状不变
    │
    ▼ RMSNorm
    稳定的表示                               ← 形状不变
    │
    ▼ LM Head (Linear)
    logits [0.1, 0.01, ..., 0.05]          ← (1, 3, vocab_size)
    │
    ▼ softmax + argmax
    预测: "sat", "on", "mat"                ← (1, 3)
```

---

## 五、和所有前续笔记的关系

这个 `TransformerLM` 是前面 9 篇笔记的**最终组装**：

| 笔记 | 组件 |
|:----|:-----|
| `03_Embedding` | `self.token_embeddings` |
| `04_RMSNorm` | `self.ln1`, `self.ln2`, `self.ln_final` |
| `02_Linear` | `self.lm_head` |
| `06_注意力机制` | `self.attn` 内部调用 `scaled_dot_product_attention` |
| `07_RoPE` | `self.attn` 中的 `RotaryPositionalEmbedding` |
| `08_多头自注意力` | `self.attn = MultiHeadSelfAttention` |
| `09_前馈网络` | `self.ffn = SwiGLU` |
| `05_Softmax` | 推理时对 logits 做 softmax 得到概率 |

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
- 所有前续笔记都在同一目录下
