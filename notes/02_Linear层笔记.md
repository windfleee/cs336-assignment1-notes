# Linear 层从零读懂

> 基于 `nn_utils.py` 中实现的 Linear 类，讲清楚四个核心问题：是什么、参数和模块、转置乘法、初始化。

---

## 一、Linear 层是什么——矩阵乘法就是加权

### 从"加权"说起

一个最简单的线性变换就是上学时学过的：

```
y = w₁·x₁ + w₂·x₂ + w₃·x₃
```

每个输入 `xᵢ` 乘一个权重 `wᵢ`，再加起来——这就是**加权求和**。权重越大，对应的输入对结果的影响就越大。

Linear 层的本质就是把这个过程**向量化、批量化**。

### 单个输出 → 多个输出

一个线性层通常有多个输出神经元（`out_features` 个），每个神经元都有自己的权重向量：

```
  [1, 3]          [3, 3]             [1, 3]

                    ┌ w₁₁  w₁₂  w₁₃ ┐
[x₁, x₂, x₃]  ×     │ w₂₁  w₂₂  w₂₃ │   =   [out₁, out₂, out₃]
  (1×3)             └ w₃₁  w₃₂  w₃₃ ┘          (1×3)
                     (3×3)
```

其中结果向量里的每个元素对应一个输出神经元的加权结果：

```
out₁ = w₁₁·x₁ + w₁₂·x₂ + w₁₃·x₃    ← 第 1 列权重对 3 个输入加权求和
out₂ = w₂₁·x₁ + w₂₂·x₂ + w₂₃·x₃    ← 第 2 列权重对 3 个输入加权求和
out₃ = w₃₁·x₁ + w₃₂·x₂ + w₃₃·x₃    ← 第 3 列权重对 3 个输入加权求和
```

**所以一个输出值 outⱼ 就是"第 j 个输出神经元对其输入做的加权和"，其中的权重 wᵢⱼ 表示"第 i 个输入对第 j 个输出有多重要"。**

用矩阵乘法写就是：

```
y = x @ Wᵀ
```

其中：
- `x` 形状 `[..., in_features]` — 输入
- `W` 形状 `[out_features, in_features]` — 权重矩阵
- `y` 形状 `[..., out_features]` — 输出


---

## 二、nn.Module 和 nn.Parameter

### nn.Module——所有"层"的爸爸

PyTorch 里，每一个神经网络层、模型、甚至是复杂的网络结构，都继承自 `nn.Module`。

```python
class Linear(nn.Module):     # ← 继承 Module
    def __init__(self, ...):
        super().__init__()   # ← 必须调用父类构造
        self.weight = nn.Parameter(...)

    def forward(self, x):    # ← 前向传播
        return x @ self.weight.T
```

`nn.Module` 给了你什么好处：

| 功能                     | 说明                                             |
| :--------------------- | :--------------------------------------------- |
| `.parameters()`        | 自动收集所有子模块和 Parameter                           |
| `.to(device)`          | 一键把整个模型搬到 GPU/CPU                              |
| `.train()` / `.eval()` | 切换训练/评估模式                                      |
| `named_children()`     | 遍历子模块                                          |
| 状态字典                   | `.state_dict()` / `.load_state_dict()` 保存和加载参数 |

**注意**：`super().__init__()` 一定要调，否则这些功能全失效。

### nn.Parameter——能被训练的"特殊张量"

```python
self.weight = nn.Parameter(torch.empty((out_features, in_features)))
```

| 对比                    | 普通 Tensor | nn.Parameter      |
| :-------------------- | :-------- | :---------------- |
| 是否被 `parameters()` 收集 | ❌         | ✅                 |
| 是否被优化器更新              | ❌         | ✅                 |
| 是否存入 `state_dict`     | ❌         | ✅                 |
| 能做的运算                 | 全部        | 全部（就是 Tensor 的子类） |
|                       |           |                   |

**一句话**：`nn.Parameter` 就是一个"贴了标签"的张量，告诉 PyTorch "这个东西要参与训练，别漏掉它"。

```python
# 错误写法：不会被优化
self.weight = torch.empty((out_features, in_features))

# 正确写法：会被优化器发现
self.weight = nn.Parameter(torch.empty((out_features, in_features)))
```

---

## 三、为什么有的是 x @ W，有的是 x @ W.T？

### 数学上没有区别

```python
y = x @ Wᵀ          # 本作业的实现
y = x @ W           # 另一种常见的写法
```

两者的数学本质完全一样——都是 `[..., in] @ [in, out]`。

区别只在于**权重矩阵 W 怎么存**：

| 写法 | 权重形状 | 前向计算 |
|:----|:--------|:--------|
| `x @ W` | `(in_features, out_features)` | 直接乘 |
| `x @ W.T` | `(out_features, in_features)` | 转置后再乘 |

### 为什么本作业选了第二种？

代码里这么写：

```python
self.weight = nn.Parameter(torch.empty((out_features, in_features)))
# 前向：x @ self.weight.T
```

原因是历史惯例和**行优先存储**。

计算机内存是"线性的"，矩阵是按**行**连续存的。`W` 的形状是 `(out_features, in_features)`，那第 `i` 行就是"第 `i` 个输出神经元的权重向量"。

```
内存里 W 是这样的：
行0: [w₀₀, w₀₁, w₀₂, ..., w₀,ᵢₙ₋₁]   ← 第 0 个输出神经元的所有权重
行1: [w₁₀, w₁₁, w₁₂, ..., w₁,ᵢₙ₋₁]   ← 第 1 个输出神经元的所有权重
...
```

当你需要做 `[..., in] @ [in, out]` 时，标准矩阵乘法要求第二个矩阵的**列**对应输出维度。所以本来的权重矩阵 `W (out, in)` 需要取**转置** `W.T (in, out)` 来乘。

两种写法的等价性：

```
x @ W.T     =     (x @ W.T)             =     y
[..., in] @ [in, out]   =   [..., out]
     ↑                    ↑
  输入 x              W 转置后 (in, out)
```

PyTorch 的 `nn.Linear` 官方实现也是存 `(out_features, in_features)` 然后用转置，和这套代码一致。

---

## 四、为什么需要特别的初始化？

### 4.1 全零初始化：加权直接失效

```python
self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
```

如果所有权重都是 0，会发生什么？

```
y = x @ 0 = 0    ← 无论输入是什么，输出全是 0
```

所有神经元的输出相同、梯度相同、更新后还是相同——整个网络**退化为一个神经元**，什么也学不到。

### 4.2 纯随机初始化：信号会炸或消失

如果只是简单随机初始化，比如 `torch.randn` 标准正态分布，问题出在**多层传播**上。

假设一个 100 层网络，每层是 `x @ W`，其中 `W ~ N(0, 1)`：

```
输入 x ~ N(0, 1)
第 1 层: x₁ = x @ W₁ → 方差 ~ 100 (因为每个输出是 100 个随机数加权)
第 2 层: x₂ = x₁ @ W₂ → 方差 ~ 10000
...

100 层后: 方差 ~ 1⁰⁰  → NaN（爆炸）
```

反过来，如果权重的方差太小，信号会逐层缩小到 0。

这就是**梯度消失/爆炸**问题——信号尺度在层间不断放大或缩小。

### 4.3 解决方案：按方差缩放初始化

关键想法是：**让每一层的输出方差 ≈ 输入方差**，这样信号经过任意多层都不炸不灭。

对于线性层 `y = x @ Wᵀ`，推导可知：

- 如果输入 `x` 的方差是 `Var(x)`，每个权重独立同分布且方差为 `Var(W)`
- 输出 `y` 的方差 ≈ `in_features × Var(W) × Var(x)`

想让 `Var(y) ≈ Var(x)`，需要：

```
Var(W) ≈ 1 / in_features
```

这就是 **Xavier/Glorot 初始化** 的核心——将权重方差设为 `1/in_features`（均匀版本）或 `2/(in_features + out_features)`（正态版本）。

### 4.4 代码中的实际做法

```python
std = (2 / (in_features + out_features)) ** 0.5
nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
```

| 参数 | 含义 |
|:----|:-----|
| `std = √(2 / (in + out))` | 标准差，来自上述方差推导 |
| `trunc_normal_` | **截断正态分布**——在 ±3σ 之外截断，避免极端值 |
| `a = -3*std, b = 3*std` | 截断边界 |

> 为什么用截断正态而不是纯正态？纯正态有小概率出现很大的初始权重，可能在一开始就让激活值爆炸。截断后保证了所有初始权重都在合理范围内。

### 一张表总结初始化

| 初始化方法 | 结果 | 原因 |
|:----------|:-----|:-----|
| 全零 | 网络学不动 | 所有神经元对称，梯度一致 |
| 全相同常数 | 同上 | 同上 |
| 纯随机 N(0,1) | 深层时炸或灭 | 方差逐层累积 |
| Xavier/trunc_normal | 信号稳定 | 方差控制在 1 附近 |

---

## 五、完整代码和思考

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return x @ self.weight.T
```

前向传播一句话：**把输入序列 `[..., in]` 通过加权变成 `[..., out]`**。这就是全连接层的核心，也是神经网络最基本的构件。

---

## 关键代码文件

- 实现代码：[nn_utils.py](/Users/justfunfun/project/cs336/assignment1-basics/cs336_basics/nn_utils.py)
