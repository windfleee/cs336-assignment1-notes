# Softmax 从零读懂

> 基于 `nn_utils.py` 中的 `softmax` 函数（没错，是个函数，不是类），
> 讲清楚它是什么、怎么算、为什么这样算。

---

## 一、Softmax 在干什么——把一组数变成概率

### 从"分数"到"概率"

假设你训练了一个分类器，输出层给了三个分数：

```
类别:      [猫,  狗,  鸟]
分数:      [2.0, 0.5, 0.1]
```

分数能看出"猫"得分最高，但你不能说"猫的概率是 2.0"——概率得在 0~1 之间，而且加起来等于 1。

**Softmax 就是干这个的**：把任意一组实数，变成一组和为 1 的概率。

### 输入输出

```
输入: [2.0, 0.5, 0.1]       ← 任意实数
                ↓
          softmax
                ↓
输出: [0.68, 0.20, 0.12]    ← 和为 1，每个值在 0~1 之间
```

| | 形状 | 含义 |
|:--|:-----|:-----|
| 输入 | `(...)` 任意形状 | 每一组待转换的实数 |
| 输出 | 和输入形状相同 | 每组值被归一化成概率 |

### 公式

$$
p_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

- 分子：对每个值取指数 $e^{x_i}$
- 分母：所有指数的和
- 结果：每个 $p_i$ 都在 0~1 之间，全部加起来等于 1

取指数有两个效果：
1. **把负数变成正数**——$e^{-100}$ 也是正数，保证概率非负
2. **放大差异**——$e^{2.0} \approx 7.4$，$e^{0.5} \approx 1.6$，分数高的被"拉开"了，差距更明显

---

## 二、代码逐行分析

### 完整代码

```python
def softmax(in_features, dim):
    maxx = torch.max(in_features, dim=dim, keepdim=True).values
    in_features_exp = torch.exp(in_features - maxx)
    in_features_exp_sum = torch.sum(in_features_exp, dim=dim, keepdim=True)
    return in_features_exp / in_features_exp_sum
```

核心就 4 行。一行一行拆。

### 第 1 行：取最大值

```python
maxx = torch.max(in_features, dim=dim, keepdim=True).values
```

| 部分 | 含义 |
|:----|:-----|
| `torch.max(x, dim=dim)` | 沿 `dim` 维取最大值 |
| `.values` | 拿到最大值（`torch.max` 返回值和索引） |
| `keepdim=True` | 保持维度数量不变，方便后面做减法 |

**为什么有 `keepdim=True`？**

```
x.shape = [B, S, V]          # 比如 B=2, S=3, V=4

torch.max(x, dim=-1)         → shape [B, S]    ← 少了 V 维
torch.max(x, dim=-1, keepdim=True) → shape [B, S, 1]  ← 保持 3 维
```

后续 `x - maxx` 需要形状匹配。`keepdim=True` 让 `maxx` 的形状变成 `[B, S, 1]`，PyTorch 自动广播成 `[B, S, V]` 再做减法。

**为什么需要这一行？**——见下面第 2 行的解释。

### 第 2 行：减最大值后取指数

```python
in_features_exp = torch.exp(in_features - maxx)
```

这行同时做了两件事：

1. **每个值减去最大值**（`in_features - maxx`）
2. **取指数**（`torch.exp(...)`）

**为什么先减最大值？**

直接套公式 $p_i = e^{x_i} / \sum e^{x_j}$，如果 $x_i$ 很大（比如 100）：

$$e^{100} \approx 2.7 \times 10^{43}$$

这个数字太大了，float32 存不下，就变成 `inf`（无穷大），后面的计算全崩。

解决方案就是**每个值先减去最大值**：

$$e^{x_i - \max(x)}$$

结果不会溢出：

```
x = [1000, 500, 200]
maxx = 1000

x - maxx = [0, -500, -700]
e^0 = 1, e^{-500} ≈ 0, e^{-700} ≈ 0
```

数学上完全等价——分子分母都除以了 $e^{\max(x)}$：

$$
\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

**这是 Softmax 实现里最重要的数值稳定技巧。没有它，大数直接炸。**

### 第 3 行：求和

```python
in_features_exp_sum = torch.sum(in_features_exp, dim=dim, keepdim=True)
```

沿指定的 `dim` 把所有指数加起来，作为概率的分母。

同样 `keepdim=True` 保证形状匹配。

### 第 4 行：归一化

```python
return in_features_exp / in_features_exp_sum
```

每个指数除以总和，得到 $[0,1]$ 之间且和为 1 的概率。

---

## 三、一个完整例子

### 手动算一遍

输入 `x = [2.0, 0.5, 0.1]`，`dim=-1`（沿最后一维）。

```
原始值:                [2.0,   0.5,   0.1]

maxx = 2.0
x - maxx:              [0.0,  -1.5,  -1.9]

exp:                   [1.0,   0.223, 0.150]
                           
sum = 1.0 + 0.223 + 0.150 = 1.373

output:                [0.728, 0.162, 0.109]
```

验证：$0.728 + 0.162 + 0.109 = 1.0$ ✅

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
- 上一篇：[04_RMSNorm层笔记.md](./04_RMSNorm层笔记.md)
- 上一篇：[02_Linear层笔记.md](./02_Linear层笔记.md) — 接在 Linear 之后用
