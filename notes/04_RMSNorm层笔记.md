# RMSNorm 从零读懂

> 基于 `nn_utils.py` 中的 RMSNorm 类，讲清楚 5 个问题：
> **为什么归一化 → 怎么做 → 沿着哪维做 → 为什么是 RMSNorm → 代码怎么跑的**

---

## 一、五个问题一览

| # | 问题 | 一句话回答 |
|:-|:-----|:----------|
| ① | **为什么要归一化？** | 防止训练过程中各层的输出均值和方差"漂移"，让梯度更新保持稳定 |
| ② | **归一化具体怎么做？** | 对一组数据，每个值减去均值、除以标准差，最后再乘一个可学习的缩放系数（必要时加偏置） |
| ③ | **沿着哪个维度归一化？** | 看场景。**BatchNorm** 沿 batch 维（第一维），**LayerNorm** 沿特征维（最后一维）；大模型用 LayerNorm |
| ④ | **大模型为什么选 RMSNorm？** | 去掉了 LayerNorm 中"减均值"的步骤，只做"除以 RMS"，计算更快、效果相当或更好 |
| ⑤ | **代码怎么跑的？** | 算均方 → 算 RMS → 归一化 → 乘权重系数，5 行核心代码 |

---

## 二、为什么要归一化？

### 训练中的"漂移"问题

假设一个网络经过多轮训练，某一层输出的分布逐渐变成这样：

```
理想状态（第 1 步）        训练一段时间后
     ↑                        ↑
     |   . . . . .            |       . . . . .
     |  . . . . . .           |    . . . . . . .
     | . . . . . . .          | . . . . . . . . . . . .
     ──────────────→          ────────────────────────→
   均值≈0, 方差≈1            均值≈2, 方差≈5
```

这就是**协变量偏移**（covariate shift）——每层的输入分布在训练过程中不断变化。

这就带来一个问题：**上一层参数更新了一点 → 输出分布变了 → 当前层看到的输入分布和当初不一样了 → 当前层又要重新适应 → 梯度更新不稳定。**

### 归一化的解决方案

每次前向传播时强行把分布拉回均值为 0、方差为 1：

```
训练一段时间后                    归一化之后
  ↑                            ↑
  |      . . . . . . . . .    |   . . . . .
  | . . . . . . . . . . . .   |  . . . . . .
  |. . . . . . . . . . . . .  | . . . . . . .
  ────────────────→            ──────────→
 均值≈2, 方差≈5               均值≈0, 方差≈1
```

这样不管上一层怎么变，当前层看到的输入始终在一个稳定的分布上，梯度更新更顺畅。

---

## 三、归一化具体怎么做？

### 标准归一化公式

给定一组数据组成一个向量 $\mathbf{x} = [x_1, x_2, \ldots, x_n]$：

$$
z_i = \frac{x_i - \mu}{\sigma + \varepsilon}
$$

其中：

| 符号 | 公式 | 含义 |
|:----|:-----|:-----|
| $\mu$ | $\displaystyle \mu = \frac{1}{n}\sum_{i=1}^{n} x_i$ | 均值 |
| $\sigma$ | $\displaystyle \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2}$ | 标准差 |
| $\varepsilon$ | 很小的常数如 $10^{-8}$ | 防止除零 |

这样出来的 $\mathbf{z}$ 就是均值为 0、方差为 1 的标准分布。

### 为了恢复学习能力：缩放 + 偏置

光归一化还不够——万一模型本来就需要某个输出的分布不是标准正态呢？

所以引入两个**可学习的参数**，让模型自己决定"归一化后应该怎么调整"：

$$
y_i = \gamma_i \cdot z_i + \beta_i
$$

- 如果模型发现"不需要调整"：$\gamma=1,\ \beta=0$，归一化结果原样输出
- 如果模型发现"需要偏移到均值为 $-1$"：$\gamma=1,\ \beta=-1$
- 甚至 $\gamma=\sigma,\ \beta=\mu$ 就完全恢复成归一化前的样子——**模型有完全的主动权**

| 参数 | 含义 | 代码里的名字 |
|:----|:-----|:------------|
| $\gamma$ (gamma) | 缩放系数 | `self.weight` |
| $\beta$ (beta) | 偏置 | RMSNorm 没有这个 |

---

## 四、沿着哪个维度归一化？

不同归一化的区别在于——**沿着哪一维算均值和方差**。

### 以形状 [B, S, d_model] 为例

```
                   batch 维      序列维       特征维
                    ↓              ↓           ↓
x.shape =         [ B,           S,         d_model ]
                   批次大小       序列长度     向量维度
```

| 归一化           | 沿哪维算             | 算个例                         | 直观理解                        |
| :------------ | :--------------- | :-------------------------- | :-------------------------- |
| **BatchNorm** | 沿 batch 维（dim=0） | 每个位置 `(s, d)` 在所有 B 个样本上算   | "同一个特征点，不同样本之间归一化"          |
| **LayerNorm** | 沿特征维（dim=-1）     | 每个元素 `(b, s)` 在 d_model 维上算 | "同一个 token，在自己内部的各维特征之间归一化" |

### 换个说法理解

拆开来看，关键就是"固定哪些维度、对哪个维度求均值/方差"：

**BatchNorm**：固定位置 $(s, d)$，对第 0 维（batch 维）做归一化。

```
输入形状 [B, S, d_model]

对每个 (s, d)   →   从所有 B 个样本中取同一位置的值   →   在这 B 个值上算 mean/std
     ↑                            ↑
 固定后两维                  第一个维度变化
```

也就是"同一个特征位置，在不同样本之间归一化"。

**LayerNorm**：固定位置 $(b, s)$，对最后一维（特征维）做归一化。

```
输入形状 [B, S, d_model]

对每个 (b, s)   →   取这一个 token 的所有 d_model 维   →   在这 d_model 个值上算 mean/std
     ↑                            ↑
 固定前两维                  最后一个维度变化
```

也就是"同一个 token，在自己的各维特征之间归一化"。



### 为什么大模型不用 BatchNorm？

两个原因：

1. **序列长度会变** — BatchNorm 需要固定 batch 维的大小，但 NLP 里句子长度不固定，LayerNorm 不关心这个
2. **训练和推理不一致** — BatchNorm 训练时用当前 batch 统计量、推理时用全局统计量；LayerNorm 训练推理一致

所以 Transformer 系列全用 **LayerNorm（沿最后一维）**。

---

## 五、大模型为什么选 RMSNorm？

### LayerNorm 的完整公式

$$
\begin{aligned}
\mu &= \text{mean}(x,\ \text{dim}=-1) \\
\sigma^2 &= \text{var}(x,\ \text{dim}=-1) \\
\hat{x} &= \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \\
y &= \gamma \cdot \hat{x} + \beta
\end{aligned}
$$

### RMSNorm 的精简

RMSNorm 去掉了**减均值**这一步，也不加偏置：

$$
\begin{aligned}
\text{RMS}(x) &= \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon} \\
\hat{x} &= \frac{x}{\text{RMS}(x)} \\
y &= \gamma \cdot \hat{x}
\end{aligned}
$$

### 省了什么？

| 对比 | LayerNorm | RMSNorm |
|:----|:---------|:--------|
| 计算均值 | ✅ 需要 | ❌ 不需要 |
| 减均值 | ✅ 需要 | ❌ 不需要 |
| 计算方差 | ✅ 需要 | ❌ 不需要 |
| 偏置参数 $\beta$ | ✅ 有 | ❌ 没有 |
| 大模型常用？ | 经典版本 | **更常用**（LLaMA、Mistral 等） |

去掉均值计算后，RMSNorm 比 LayerNorm **快 5%~10%**，且在语言建模任务上效果相当甚至更好。对于几十亿参数的大模型，这点加速很可观。

---

## 六、结合代码分析 RMSNorm

### 完整代码

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape  # d_model 的大小
        self.weight = nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype)
        )   # 可学习的缩放系数 gamma，初始全 1
        self.eps = eps

    def forward(self, x):
        xdtype = x.dtype
        x = x.to(torch.float32)                       # ① 转 float32 保证精度

        mean_square = x.pow(2).mean(dim=-1, keepdim=True)  # ② 算均方
        rms = (mean_square + self.eps).sqrt()              # ③ 算 RMS

        x_norm = x / rms * self.weight                     # ④ 归一化 + 缩放

        return x_norm.to(xdtype)                           # ⑤ 转回原类型
```

### 逐行拆解

**① 转 float32**

```python
x = x.to(torch.float32)
```

归一化涉及除法和开方，用 float32 做更稳定。输入可能是 float16（为了省显存），但归一化时转到 float32 算，算完再转回去。

**② 算均方（mean of squares）**

```python
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
```

`x.pow(2)` — 每个元素平方。

`.mean(dim=-1)` — 沿**最后一维**（单 token 的 d_model 维）求平均值。

`.keepdim=True` — 保持维度数量，方便后面除法的 shape 匹配。

```
x.shape = [B, S, d_model]

x.pow(2).shape           = [B, S, d_model]
  .mean(dim=-1)          = [B, S]            ← keepdim=False 的话
  .mean(dim=-1, keepdim) = [B, S, 1]         ← keepdim=True
```

`keepdim=True` 让结果形状 $[\text{B}, \text{S}, 1]$，后续 `x / rms` 时 PyTorch 会自动广播（broadcast）成 $[\text{B}, \text{S}, d_{\text{model}}]$。

**③ 算 RMS（Root Mean Square）**

```python
rms = (mean_square + self.eps).sqrt()
```

加 `eps` 防止 `mean_square` 为 0 时除零（比如输入全 0 的情况）。

RMS 的含义就是**均方根**——每个值的平方的平均数的平方根。

**④ 归一化 + 缩放**

```python
x_norm = x / rms * self.weight
```

$\displaystyle \text{RMS} = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}$，先对每个 token 向量除以自己的 RMS，然后把整个向量逐元素乘上 `self.weight`（形状是 $(d,)$，自动广播）。

| 步骤 | 运算 | 形状变化 |
|:----|:-----|:---------|
| 输入 x | — | $[\text{B}, \text{S}, d]$ |
| 除以 RMS | $x / \text{RMS}$ | $[\text{B}, \text{S}, d]$（广播后） |
| 乘权重 | $\cdot\ \text{weight}$ | $[\text{B}, \text{S}, d]$（广播后） |

**⑤ 转回原类型**

```python
return x_norm.to(xdtype)
```

如果输入是 float16，输出也转回 float16，方便后续计算。

### 一个完整例子

```python
# 输入：B=2, S=3, d_model=4
x = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 4.0, 6.0, 8.0],
     [1.0, 1.0, 1.0, 1.0]],
    [[0.0, 0.0, 0.0, 1.0],
     [5.0, 5.0, 5.0, 5.0],
     [0.0, 1.0, 2.0, 3.0]],
])

rmsnorm = RMSNorm(normalized_shape=4)
output = rmsnorm(x)
```

第一步——均方：

```python
x.pow(2) = [[[1, 4, 9, 16], [4, 16, 36, 64], [1, 1, 1, 1]],
            [[0, 0, 0, 1],  [25, 25, 25, 25], [0, 1, 4, 9]]]

mean_square = x.pow(2).mean(dim=-1, keepdim=True)
           = [[[7.5], [30], [1]],
              [[0.25], [25], [3.5]]]       # 形状 [2, 3, 1]
```

第二步——RMS：

```python
rms = sqrt(mean_square + 1e-8)
    ≈ [[[2.74], [5.48], [1.0]],
       [[0.5],  [5.0],  [1.87]]]           # 形状 [2, 3, 1]
```

第三步——归一化：

```python
x / rms  ≈ [[[0.36, 0.73, 1.09, 1.46],
              [0.36, 0.73, 1.09, 1.46],
              [1.0,  1.0,  1.0,  1.0]],
             [[0.0,  0.0,  0.0,  2.0],
              [1.0,  1.0,  1.0,  1.0],
              [0.0,  0.53, 1.07, 1.60]]]
```

注意每个 token 向量的**长度（RMS）被归一化到了 1 左右**，但向量之间的相对比例保留了。再乘上 `self.weight`（初始全 1）后输出不变——随着训练推进，`weight` 会学习到合适的缩放。

---

## 七、和其他层的对比

| 层 | 做什么 | 有无可学习参数 | 初始化 |
|:--|:-------|:-------------|:-------|
| **Linear** | 加权融合 | `weight` 形状 $(out, in)$ | trunc_normal，std 按维度缩放 |
| **Embedding** | 查表映射 | `weight` 形状 $(V, d)$ | trunc_normal，std=1 |
| **RMSNorm** | 归一化稳定分布 | `weight` 形状 $(d,)$ | 全 1（不做任何缩放） |

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
- 上一篇：[03_Embedding层笔记.md](./03_Embedding层笔记.md)
- 上一篇：[02_Linear层笔记.md](./02_Linear层笔记.md)
