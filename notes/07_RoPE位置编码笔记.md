# RoPE（旋转位置编码）从零读懂

> 基于 `nn_utils.py` 中的 `RotaryPositionalEmbedding` 类，
> 讲清楚三个问题：**为什么需要 RoPE → 二维旋转的复数理解 → 代码逐行拆解**

---
![[Pasted image 20260724153007.png]]
## 一、为什么要 RoPE？——绝对位置编码的局限

### 绝对位置编码的问题

一种朴素的想法是：给每个位置分配一个唯一的向量，加到 embedding 上。

```python
# 绝对位置编码的简化思路
h = token_embedding + pos_encoding(positions)    # 一次注入，简单相加
# 后续各层：位置信息靠残差流传递，越往后占比越小
```

有两个核心劣势：

**劣势一：一次注入，被稀释**

位置信息只在输入层加一次，后续靠残差流向下传递。随着层数增加，残差流中叠加了大量内容变换，原始位置信号的占比越来越小——位置信息被逐渐"冲淡"。

```
层 1:  h₁ = Attention(Q₁, K₁, V₁) + (token + pos)    ← pos 清晰
层 5:  h₅ = ... + (token + pos + 前 4 层输出)           ← pos 占比变小
层 20: h₂₀ = ... + (大量内容混合 + 微量原始 pos)         ← pos 被淹没
```

**劣势二：不具有长度外推性**

绝对位置编码的矩阵大小是预设的（比如 512 x 768），训练时模型只见过 0~511 的位置向量。

如果想把输入长度扩展到 1024，直接做法是把矩阵拼成 1024 x 768——但新添的 512 个位置向量完全没有训练过，它们对应的位置编码无法正确表示 512~1023 的位置信息。

```
训练时:  位置 0~511 的向量已充分训练
推理时:  [位置 0~511 的表现正常 | 位置 512~1023 胡乱输出]
          ↑ 见过              ↑ 没见过，随机初始化或插值，效果差
```

扩展位置矩阵会**破坏模型在预训练阶段学到的位置信息分布**——这不是简单地"加几行"就能解决的。

> 不过早期对长文本的需求并不迫切（BERT 最大 512，GPT-2 最大 1024），所以这个缺陷在当时不是致命问题。

**劣势三：反应不了相对位置**

Attention 的点积真正需要的是"相隔几个位置"的相对信息。

但 APE 的点积展开式里只有绝对位置项，没有天然出现 $(p - q)$：

$$
Q_p \cdot K_q = f(v_p, v_q, p, q) \quad \text{(含 }p\text{ 和 }q\text{，但不含 }p-q\text{)}
$$

模型需要靠大量数据和参数去**硬学**这个映射——"位置 2 和位置 5 的关系 ≈ 位置 3 和位置 6 的关系"。

RoPE 直接通过旋转把 $(p - q)$ 编码到点积公式里，模型不需要学这个。

### 相对位置才是核心

一个好的位置编码，核心是让模型能知道：**"词 A 和词 B 之间隔了几个词"**。

```
"The cat sat on the mat"

  猫 和 垫 之间隔了 4 个词 → 这种"相对距离"比"猫在第 1 位，垫在第 5 位"更关键
```

### RoPE 的核心思想

RoPE 不把位置加到 embedding 上，而是**直接旋转 Q 和 K 向量**，让 Q 和 K 的点积**天然包含它们之间的相对位置信息**。

```
标准 Attention:  Q · K          ← 只衡量"内容相似度"

RoPE Attention:  RoPE(Q) · RoPE(K)   ← 同时衡量"内容 + 距离"
```

具体来说，RoPE 通过旋转使得：

$$
\text{RoPE}(Q_p) \cdot \text{RoPE}(K_q) = f(Q, K, p - q)
$$

即 Q 在位置 p、K 在位置 q 的点积，只取决于 Q、K 的内容和它们的**相对距离** $p - q$，与绝对位置 $p$ 和 $q$ 无关。
![[Pasted image 20260724153752.png]]

---

## 二、二维向量的旋转——从复数域理解

### 核心操作：旋转一个二维向量

在二维平面上，将一个向量 $(x_0, x_1)$ 旋转角度 $\theta$：

$$
\begin{pmatrix}
x_0' \\
x_1'
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
x_0 \\
x_1
\end{pmatrix}
$$

展开就是：

$$
\begin{aligned}
x_0' &= x_0 \cdot \cos\theta - x_1 \cdot \sin\theta \\
x_1' &= x_0 \cdot \sin\theta + x_1 \cdot \cos\theta
\end{aligned}
$$

角度 $\theta$ 越大，向量转得越多。不同位置用不同角度，向量就指向不同方向。

### 复数域的理解

二维向量 $(x_0, x_1)$ 可以看作一个复数 $z = x_0 + i x_1$。

旋转角度 $\theta$ 等价于**乘以单位复数** $e^{i\theta}$：

$$
z' = z \cdot e^{i\theta} = (x_0 + i x_1)(\cos\theta + i\sin\theta)
$$

展开：

$$
\begin{aligned}
z' &= x_0\cos\theta - x_1\sin\theta + i(x_0\sin\theta + x_1\cos\theta) \\
  &= x_0' + i x_1'
\end{aligned}
$$

和上面的矩阵形式完全等价。

| 表示 | 公式 | 对应代码 |
|:----|:-----|:---------|
| 矩阵 | $\begin{pmatrix}x_0'\\x_1'\end{pmatrix} = \begin{pmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{pmatrix}\begin{pmatrix}x_0\\x_1\end{pmatrix}$ | `x0*cos - x1*sin`, `x0*sin + x1*cos` |
| 复数 | $z' = z \cdot e^{i\theta}$ | 理解本质更容易 |

**旋转的关键性质**：两个旋转后的向量做点积，结果只和它们的**角度差**有关：

$$
\text{Rot}(v_p) \cdot \text{Rot}(v_q) = v_p \cdot v_q \cdot \cos(\theta_p - \theta_q)
$$

这正是 RoPE 让点积蕴含相对位置信息的数学根源。

### 从 2D 推广到 d_k 维

d_k 维向量（比如 d_k=64）不能一次旋转完。RoPE 的做法是**两两分组**：

```
d_k = 64 维的 Q 向量:

[对0-1  |  对2-3  |  对4-5  |  ...  |  对62-63]
   ↓         ↓         ↓                ↓
旋转θ₁     旋转θ₂     旋转θ₃           旋转θ₃₂
```

每对维度有自己独立的旋转频率：

| 维度对 | 旋转角速度 |
|:------|:----------|
| (0, 1) | $\theta_0 = 1$（最快） |
| (2, 3) | $\theta_1 = \Theta^{-2/d_k}$ |
| (4, 5) | $\theta_2 = \Theta^{-4/d_k}$ |
| ... | ... |
| (d_k-2, d_k-1) | $\theta_{d_k/2-1} = \Theta^{-(d_k-2)/d_k}$（最慢） |

其中 $\Theta$ 是一个超参数（代码里叫 `theta`，默认常取 10000）。

公式统一写成：

$$
\theta_i = \Theta^{-2i/d_k} \quad \text{其中}\ i = 0, 1, \ldots, d_k/2-1
$$

### 为什么用多个频率？

低频（高 i）对旋转不敏感 → 捕捉长距离依赖
高频（低 i）对旋转敏感 → 捕捉短距离关系

这和正弦位置编码的直觉一致——不同频率的三角函数组合起来，可以编码不同粒度的位置信息。

---

## 三、逐行代码解释

### 3.1 `__init__`：预计算 cos/sin 表

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
```

保存参数，后面 forward 要用。

#### 第 1 步：计算频率向量

```python
i = torch.arange(d_k // 2, device=device)          # (d_k//2,)
freqs = 1.0 / (theta ** (2 * i / d_k)).reshape(1, d_k // 2)
```

- `i = [0, 1, 2, ..., d_k//2 - 1]`
- `freqs[i] = 1 / theta ** (2i / d_k)` —— 这就是每对维度的旋转频率

```
假设 d_k=4:
i = [0, 1]
freqs = [1 / theta^0, 1 / theta^(2/4)]
       = [1, 1/√theta]
```

`.reshape(1, d_k//2)` 加了一个维度，为后面矩阵乘法做准备。

#### 第 2 步：位置 × 频率 = 每个位置每个频率的旋转角度

```python
positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
           .reshape(max_seq_len, 1)          # (max_seq_len, 1)
angle = positions @ freqs                     # (max_seq_len, d_k//2)
```

矩阵乘法：

```
angle[pos][i] = pos × freqs[i] = pos / Theta^(2i/d_k)
```

```
                 freqs[0]  freqs[1]  ...
          ┌────────────────────────────
pos=0     │   0        0        ...
pos=1     │  freqs[0]  freqs[1]  ...
pos=2     │ 2*freqs[0] 2*freqs[1] ...
  ...     │
```

`angle[pos][i]` 就是位置 `pos` 在第 `i` 对维度上的旋转角度。

#### 第 3 步：计算 cos 和 sin

```python
cos_cached = torch.cos(angle)    # (max_seq_len, d_k//2)
sin_cached = torch.sin(angle)    # (max_seq_len, d_k//2)
```

每个位置、每对维度都有对应的 cos/sin。

注意这里形状是 `(max_seq_len, d_k//2)`，还差一步扩展到 `d_k`。

#### 第 4 步：注册为 buffer（保存在模型里但不参与训练）

```python
self.register_buffer("cos_cached", cos_cached)  # (max_seq_len, d_k//2)
self.register_buffer("sin_cached", sin_cached)
```

`register_buffer` 意味着这些张量：
- 保存在模型里（`state_dict` 包含它们）
- 不参与梯度更新
- 自动搬到 device

> 注意：这里存的 cos/sin 形状是 `(max_seq_len, d_k//2)`，不是 `(max_seq_len, d_k)`。
> 到 forward 里索引后才会用 `view` 展开到 `d_k`——见下面解释。

---

### 3.2 `forward`：对输入施加旋转

```python
def forward(self, x, token_positions):
```

| 参数 | 形状 | 含义 |
|:----|:-----|:-----|
| `x` | `(..., seq_len, d_k)` | Q 或 K 向量 |
| `token_positions` | `(..., seq_len)` | 每个 token 的位置索引 |

#### 第 1 步：根据位置取出对应的 cos/sin

```python
cos = self.cos_cached[token_positions]    # (..., seq_len, d_k//2)
sin = self.sin_cached[token_positions]    # (..., seq_len, d_k//2)
```

`token_positions` 是整数张量，用花式索引从预计算表中取对应行的 cos/sin。

但这里取出来的形状是 `(..., seq_len, d_k//2)`，而输入 x 的形状是 `(..., seq_len, d_k)`。差了个因子 2。

> **为什么存的是 `(d_k//2)` 而不是 `(d_k)`？**
>
> 因为要旋转的是**连续两维**: `(x[2i], x[2i+1])`。存了 `d_k//2` 个 cos 值，每个 cos 管一对维度。

#### 第 2 步：reshape 成对，施加旋转

```python
x_reshaped = x.view(*x.shape[:-1], self.d_k // 2, 2)
```

把最后一维 `d_k` reshape 成 `(d_k//2, 2)`：

```
原始 x:  [a0, a1, b0, b1, c0, c1, ...]      ← d_k 个值
                    ↓ view
        [(a0, a1), (b0, b1), (c0, c1), ...]  ← d_k//2 对
```

```python
x_rotated = torch.empty_like(x_reshaped)
x_rotated[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
x_rotated[..., 1] = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
```

对每对维度 `(x[0], x[1])` 施加二维旋转矩阵：

$$
\begin{pmatrix}
x_0' \\
x_1'
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix}
x_0 \\
x_1
\end{pmatrix}
$$

复数的理解：$z' = z \cdot e^{i\theta}$

注意这里 `cos` 和 `sin` 的形状是 `(..., seq_len, d_k//2)`，正好和 `x_reshaped[..., 0]` 形状 `(..., seq_len, d_k//2)` 匹配——逐元素乘法。

**这个过程可以用一个更紧凑的图表示：**

```
每对维度                          旋转后
┌──────┐                         ┌──────┐
│ x[0] │──┐  ┌─────────────────┐ │ x'[0]│
│ x[1] │──┼──│  x[0]*cos - x[1]*sin │ x'[1]│
├──────┤  │  │  x[0]*sin + x[1]*cos │──────┤
│ x[2] │──┼──│                 │ │ x'[2]│
│ x[3] │──┘  └─────────────────┘ │ x'[3]│
├──────┤                          ├──────┤
│ ...  │         ...              │ ...  │
└──────┘                          └──────┘
```

#### 第 3 步：reshape 回原始形状

```python
return x_rotated.view_as(x)
```

把 `(..., seq_len, d_k//2, 2)` 还原回 `(..., seq_len, d_k)`。

---

## 四、RoPE 在 Attention 中的位置

```
输入 x: (B, S, d_model)
    │
    ├──→ Q = Linear(x)          ← 从 x 投影出 Q
    ├──→ K = Linear(x)          ← 从 x 投影出 K
    └──→ V = Linear(x)          ← 从 x 投影出 V
    │
    ├──→ Q = RoPE(Q, positions)  ← 对 Q 施加旋转位置编码
    ├──→ K = RoPE(K, positions)  ← 对 K 施加旋转位置编码
    │
    └──→ Attention(Q, K, V)     ← Q 和 K 的点积已含相对位置信息
```

RoPE 加在 **Q 和 K 投影之后、Attention 计算之前**。V 不需要旋转，因为 V 只参与加权求和，不参与相似度计算。

---

## 五、和之前各层的对比

| 层/函数 | 做的事 | 有无可学习参数 | 备注 |
|:--------|:-------|:-------------|:-----|
| **Linear** | 加权变换 | ✅ `weight` | QKV 由它投影 |
| **Embedding** | 查表映射 | ✅ `weight` | token ID → 向量 |
| **RMSNorm** | 归一化 | ✅ `weight` | 稳定分布 |
| **Softmax** | 分数→概率 | ❌ | 纯函数 |
| **Attention** | 相似度加权 | ❌ | QKV 来自 Linear |
| **RoPE** | 旋转编码位置 | ❌ | 纯计算，cos/sin 预计算后冻结 |

RoPE 没有可学习参数——cos 和 sin 的缓存表是**固定计算**出来的，`register_buffer` 不参与梯度更新。

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
- 上一篇：[06_注意力机制笔记.md](./06_注意力机制笔记.md) — RoPE 的旋转结果喂给 Attention
- 上一篇：[02_Linear层笔记.md](./02_Linear层笔记.md) — QKV 由 Linear 投影
