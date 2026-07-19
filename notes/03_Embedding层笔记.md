# Embedding 层从零读懂

> 基于 `nn_utils.py` 中的 Embedding 类，理清文本向量化的最后一块拼图。

---
![[Pasted image 20260719155224.png]]
## 一、Embedding 在干什么——把 token ID 变成向量

### 整条流水线回顾

之前的两份笔记各自覆盖了一段链路：

```
文本 "hello world"
    │
    ▼ (tokenizer)
[1523, 42, 7, 103, ...]          ← token ID 序列（整数）
    │
    ▼ (embedding)                 ← 你在这里
[[0.12, -0.34, ...],
 [0.55,  0.21, ...],             ← token 向量序列（浮点数）
 [0.03, -0.78, ...],             ← 每个 token 变成 d 维向量
 ...]
    │
    ▼ (Transformer)
模型内部计算
```

**Tokenizer** 把文本变成整数 ID，**Embedding** 把这些整数 ID 变成模型能算的浮点数向量。

### Embedding 的输入和输出

```
输入: token_ids, 形状 [B, S]          ← S 个 token，每个存一个整数 ID
                    ↓
         每个 ID 去查一张大表
                    ↓
输出: vectors,    形状 [B, S, d]      ← 每个 token ID 换成一个 d 维向量
```

| | 形状 | 含义 | 值的样子 |
|:--|:-----|:-----|:---------|
| 输入 | `(B, S)` 或 `(S,)` | batch 大小 × 序列长度 | `[42, 7, 103]` |
| 输出 | `(B, S, d)` 或 `(S, d)` | batch 大小 × 序列长度 × 向量维度 | `[[0.12, -0.34, ...], ...]` |

> 严格来说输入形状是 `(...)` 任意维度，输出形状在末尾多一个 `d`。即 `(...)` → `(..., d)`。

---

## 二、Embedding 的代码结构

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        std = 1
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

`__init__` 创建一张**可训练的查找表**——就是一个矩阵 `weight`，形状 `(V, d)`：

| 维度                 | 含义                | 例子        |
| :----------------- | :---------------- | :-------- |
| V (num_embeddings) | 词汇表大小，ID 的取值范围    | V = 50257 |
| d (embedding_dim)  | 每个 token 被映射成几维向量 | d = 768   |
|                    |                   |           |

**第 `i` 行就是 token ID = `i` 对应的那个向量。**

```
        d 列
    ┌────────────────┐
  0  │  向量 0         │    ← token ID=0 对应的向量
  1  │  向量 1         │    ← token ID=1 对应的向量
V  2  │  向量 2         │    ← token ID=2 对应的向量
行  ...│  ...            │
V-1  │  向量 V-1       │    ← token ID=V-1 对应的向量
    └────────────────┘
```

---

## 三、花式索引——`self.weight[token_ids]` 是怎么工作的？

这是 Embedding 最核心的一行代码，理解了它，整个层就懂了。

### 普通索引 vs 花式索引

```python
# 普通索引：取某一行
self.weight[2]           # → 形状 [d]，取第 2 行（token ID=2 的向量）

# 花式索引（fancy indexing）：用一个整数张量去取多行
token_ids = torch.tensor([2, 5, 2])
self.weight[token_ids]   # → 形状 [3, d]
                         #    取第 2 行、第 5 行、第 2 行
```

花式索引的结果形状 = **索引张量的形状** + **被索引张量去掉被索引那维后的形状**。

### 具体看

假设词汇表大小 V=6（6 个词），embedding 维度 d=4：

```python
weight = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],     # ID=0: "the"
    [0.5, 0.6, 0.7, 0.8],     # ID=1: "cat"
    [0.9, 1.0, 1.1, 1.2],     # ID=2: "sat"
    [1.3, 1.4, 1.5, 1.6],     # ID=3: "on"
    [1.7, 1.8, 1.9, 2.0],     # ID=4: "the"
    [2.1, 2.2, 2.3, 2.4],     # ID=5: "mat"
])
# weight.shape = (6, 4)

token_ids = torch.tensor([2, 5, 2])          # "sat", "mat", "sat"
# token_ids.shape = (3,)

result = weight[token_ids]                   # → shape (3, 4)
# result = [
#   [0.9, 1.0, 1.1, 1.2],   ← weight[2], 对应 "sat"
#   [2.1, 2.2, 2.3, 2.4],   ← weight[5], 对应 "mat"
#   [0.9, 1.0, 1.1, 1.2],   ← weight[2], 对应 "sat"
# ]
```

### 直觉理解

```
                    weight: (V=6, d=4)
              ┌──────────────────────────┐
token_ids     │  0: [0.1, 0.2, 0.3, 0.4] │
  [2]  ───────→  1: [0.5, 0.6, 0.7, 0.8] │────→ [0.9, 1.0, 1.1, 1.2]
  [5]  ───────→  2: [0.9, 1.0, 1.1, 1.2] │────→ [2.1, 2.2, 2.3, 2.4]
  [2]  ───────→  3: [1.3, 1.4, 1.5, 1.6] │────→ [0.9, 1.0, 1.1, 1.2]
                4: [1.7, 1.8, 1.9, 2.0] │
                5: [2.1, 2.2, 2.3, 2.4] │
              └──────────────────────────┘
```

每个 token ID 就像一把钥匙，去 `weight` 这张表里抽出对应的那一行。**一模一样 ID 就取同一行**（ID=2 出现了两次，结果里对应两行一模一样的向量）。

---

## 四、为什么 std = 1？

### 和 Linear 对比

| 层         | 初始化 std                   | 公式             |
| :-------- | :------------------------ | :------------- |
| Linear    | `std = √(2 / (in + out))` | 防止信号经过矩阵乘法后炸/灭 |
| Embedding | `std = 1`                 | 希望输出的方差是 1     |
|           |                           |                |

### Linear 需要缩放的原因

Linear 是 `y = x @ Wᵀ`——输入和权重做**矩阵乘法**。每个输出元素是 `in_features` 个乘积的和：

```
outⱼ = x₁·w₁ⱼ + x₂·w₂ⱼ + ... + xᵢₙ·wᵢₙ,ⱼ
```

如果每个乘积不缩放到合适大小，和的结果方差会随着 `in_features` 变大而爆炸。

### Embedding 不需要缩放到

Embedding 是 `y = weight[id]`——只是一个**查表操作**，没有加法累加。

```
          ┌── 直接取出来，没有任何加法 ──┐
y = weight[token_id]       ← 只是取一行出来
```

所以 Embedding 不需要像 Linear 那样按维度缩放。那为什么不用 `std=1` 之外的数？

因为**后续的模型层期望输入的方差大概在 1 附近**。如果你把 embedding 初始化成 `std=0.01`，所有向量挤在一个小圆点里，信息量太小；如果 `std=100`，向量炸得到处都是，后续层直接 NaN。

**`std=1` 是一个"中性"的出发点**——让每个 token 在向量空间中均匀散布，方差和后续层的预期一致，随着训练推进参数会根据任务自动调整。

---

## 五、一句话总结

| 问题 | 回答 |
|:----|:-----|
| 它在干什么？ | 把 token ID（整数）映射成可训练的稠密向量 |
| 怎么做的？ | `self.weight[token_ids]`——花式索引查表 |
| 和 Linear 什么区别？ | 没有加权计算，没有加法累加，只是取一行 |
| 为什么 std=1？ | 因为是纯查表，不需要缩放；输出方差 ≈ 1 和后续层预期一致 |

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
- 前一篇：[01_BPE_Tokenizer笔记.md](./01_BPE_Tokenizer笔记.md) — 文本 → token ID
- 前一篇：[02_Linear层笔记.md](./02_Linear层笔记.md) — 加权变换
