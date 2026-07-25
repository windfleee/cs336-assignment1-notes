from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
from torch import Tensor



class Linear(nn.Module):
    """Linear transformation layer (no bias).

    Performs y = x @ W^T.
    """

    def __init__(
        self,
        in_features: int,    # 输入特征的维度（最后一维的大小）
        out_features: int,   # 输出特征的维度（最后一维的大小）
        device: torch.device | None = None,  # 参数存储的设备（如 'cpu', 'cuda:0'）
        dtype: torch.dtype | None = None,    # 参数的数据类型（如 torch.float32）
    ):
        super().__init__()
        # TODO: 创建 self.weight = nn.Parameter(...)
        # - 形状: (out_features, in_features)
        # - 用 torch.nn.init.trunc_normal_ 初始化
        # - 注意 device 和 dtype 参数
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a= -3 * std, b= 3 * std)
    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        # TODO: 实现 y = x @ W^T
        #[... in_features] @ [in_features , out_features ] -> [... out_features]
        return x @ self.weight.T


class Embedding(nn.Module):
    """Embedding lookup layer.

    Given a tensor of integer token IDs, returns the corresponding embedding
    vectors from a learnable embedding matrix.
    """
    def __init__(
        self,
        num_embeddings: int,       # 词汇表大小
        embedding_dim: int,        # embedding 向量的维度，即 d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # TODO: 创建 self.weight = nn.Parameter(...)
        # - 形状: (num_embeddings, embedding_dim)
        # - 使用 torch.empty(...)，传入 device 和 dtype
        # - 用 torch.nn.init.trunc_normal_ 初始化权重
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        std = 1
        nn.init.trunc_normal_(self.weight, std=std, a= -3 * std, b= 3 * std)
    def forward(
        self, token_ids: Int[Tensor, "..."]
    ) -> Float[Tensor, "... embedding_dim"]:
        # TODO: 实现 embedding 查表
        # - 用 token_ids 对 self.weight 做索引
        # - 输入形状 (...) → 输出形状 (..., embedding_dim)
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    This is a simplified version of LayerNorm that only normalizes by the root mean square
    of the input, without centering or scaling by learned parameters.
    """
    def __init__(
        self,
        normalized_shape: int,  # 输入特征的维度（最后一维的大小）
        eps: float = 1e-8,      # 防止除零的小常数
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        #初始化权重参数
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.eps = eps
        
    def forward(self, x: Float[Tensor, "... normalized_shape"]) -> Float[Tensor, "... normalized_shape"]:
        # TODO: 实现 RMSNorm
        xdtype = x.dtype
        x = x.to(torch.float32)  # 为了数值稳定性，先转换为 float32
        #最后一维算平均
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        #计算rms
        rms = (mean_square + self.eps).sqrt()
        #乘上缩放系数
        x_norm = x / rms * self.weight

        return x_norm.to(xdtype)  # 转回原来的数据类型

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    #取dim维度的最大值
    maxx = torch.max(in_features,dim=dim,keepdim=True).values
    #减去最大值后，取exp
    in_features_exp = torch.exp(in_features - maxx)
    #求和
    in_features_exp_sum = torch.sum(in_features_exp,dim = dim,keepdim = True)
    #按元素相除
    return in_features_exp / in_features_exp_sum

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given query, key, and value tensors, compute the scaled dot product attention.

    Args:
        Q (Float[Tensor, "... queries d_k"]): Query tensor.
        K (Float[Tensor, "... keys d_k"]): Key tensor.
        V (Float[Tensor, "... keys d_v"]): Value tensor.
        mask (Bool[Tensor, "... queries keys"] | None): Optional mask tensor. If provided,
            it should be broadcastable to the shape of the attention scores.

    Returns:
        Float[Tensor, "... queries d_v"]: The result of applying scaled dot product attention.
    """
    # Step 1: 计算 scaled attention scores
    # Q:      (..., queries, d_k)
    # K^T:    (..., d_k, keys)
    # scores: (..., queries, keys)
    qkt = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)

    # Step 2: 应用 mask（True=保留, False→-inf）
    if mask is not None:
        qkt = qkt.masked_fill(mask == False, float('-inf'))

    # Step 3: softmax 沿 keys 维度归一化 → (..., queries, keys)
    scaled_attention = softmax(qkt, dim=-1)

    # Step 4: 加权聚合 values
    # attn:   (..., queries, keys)
    # V:      (..., keys, d_v)
    # output: (..., queries, d_v)
    return scaled_attention @ V


class RotaryPositionalEmbedding(nn.Module):
    """Applies Rotary Position Embedding (RoPE) to query or key tensors.

    RoPE encodes position information by rotating pairs of dimensions
    in the input tensor according to the token's position.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Construct the RoPE module and precompute cos/sin buffers.

        Args:
            theta: Θ value controlling the base rotation frequency.
            d_k: Dimension of query/key vectors (must be even).
            max_seq_len: Maximum sequence length to precompute for.
            device: Device for the precomputed buffers.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # TODO: 1. 计算频率向量 freqs[i] = 1.0 / (theta ** (2*i / d_k))，形状 (d_k // 2,)
        i = torch.arange(d_k // 2, device=device)  # (d_k // 2,)
        freqs = 1.0 / (theta ** (2 * i / d_k    )).reshape(1, d_k // 2)  # (d_k // 2,)
        # TODO: 2. 计算所有位置的角度矩阵 positions @ freqs，形状 (max_seq_len, d_k // 2)
        positions = torch.arange(max_seq_len,device=device,dtype=torch.float32).reshape(max_seq_len, 1)  # (max_seq_len, 1)
        angle = positions @ freqs  # (max_seq_len, d_k // 2) 
        # TODO: 3. 计算 cos 和 sin，每个角度重复一次以匹配 d_k 维度
        cos_cached = torch.cos(angle)  # (max_seq_len, d_k//2)
        sin_cached = torch.sin(angle)  # (max_seq_len, d_k//2)
        # TODO: 4. 用 register_buffer 保存 cos_cached 和 sin_cached，形状均为 (max_seq_len, d_k)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Apply RoPE to the input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k).
            token_positions: Token position indices of shape (..., seq_len).

        Returns:
            Tensor of same shape as x with RoPE applied.
        """
        # TODO: 1. 用 token_positions 从 cos_cached / sin_cached 中取出对应位置的旋转系数
        #       得到形状 (..., seq_len, d_k) 的 cos 和 sin
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k)
        # TODO: 2. 将 x reshape 为 (..., seq_len, d_k//2, 2)，对每对维度施加旋转
        x_reshaped = x.view(*x.shape[:-1], self.d_k // 2, 2)  # (..., seq_len, d_k//2, 2)
        x_rotated = torch.empty_like(x_reshaped)
        x_rotated[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
        x_rotated[..., 1] = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos   
        # TODO: 3. reshape 回原始形状并返回
        return x_rotated.view_as(x)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Supports optional Rotary Position Embedding (RoPE).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        # TODO: 创建四个投影层 q_proj, k_proj, v_proj, o_proj（用 Linear 类）
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # TODO: 如果 use_rope=True，创建 RotaryPositionalEmbedding 实例
        #       需要 theta, d_k, max_seq_len 参数
        if use_rope:
            assert theta is not None, "theta must be provided if use_rope is True"
            assert max_seq_len is not None, "max_seq_len must be provided if use_rope is True"
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        # TODO: 1. 投影 Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)  # (..., seq_len, d_model)
        V = self.v_proj(x)  # (..., seq_len, d_model)
        # TODO: 2. 拆分多头 (reshape + transpose)
        # Q, K, V: (..., seq_len, d_model) → (..., seq_len, num_heads, d_k) → (..., num_heads, seq_len, d_k)
        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (..., num_heads, seq_len, d_k)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (..., num_heads, seq_len, d_k)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)  # (..., num_heads, seq_len, d_k)
        # TODO: 3. 如果 use_rope，对 Q 和 K 应用 RoPE
        if self.use_rope:
            assert token_positions is not None, "token_positions must be provided if use_rope is True"
            Q = self.rope(Q, token_positions)  # (..., num_heads, seq_len, d_k)
            K = self.rope(K, token_positions)  # (..., num_heads, seq_len, d_k)
        # TODO: 4. 构建因果 mask（下三角布尔矩阵）
        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), device=x.device, dtype=torch.bool))  # (seq_len, seq_len)
        # TODO: 5. 调用 scaled_dot_product_attention(Q, K, V, mask)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)  # (..., num_heads, seq_len, d_k)
        # TODO: 6. 合并多头 (transpose + reshape)
        res = attn_output.transpose(-3, -2).reshape(*attn_output.shape[:-3], x.shape[-2], self.d_model)  # (..., seq_len, d_model)
        # TODO: 7. 输出投影
        res = self.o_proj(res)  # (..., seq_len, d_model)
        return res


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Apply SiLU activation element-wise.

    SiLU(x) = x * sigmoid(x)
    """
    # TODO: 实现 SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T
    """

    def __init__(
        self,
        d_model: int,    # 输入/输出维度
        d_ff: int,       # 内部 FFN 维度
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # TODO: 创建三个 Linear 层
        # W1: (d_ff, d_model) — gate 通路
        # W2: (d_model, d_ff) — 输出投影
        # W3: (d_ff, d_model) — up 通路
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear(d_model,d_ff,device=device, dtype=dtype) 
        self.w2 = Linear(d_ff,d_model, device=device, dtype=dtype)  # TODO
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # TODO
    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        # TODO: 实现 SwiGLU(x) = (SiLU(x @ W1^T) * (x @ W3^T)) @ W2^T
        # Step 1: gate = silu(x @ W1^T)   → (..., d_ff)
        # Step 2: up   = x @ W3^T         → (..., d_ff)
        # Step 3: output = (gate * up) @ W2^T  → (..., d_model)
        return self.w2(silu(self.w1(x)) * self.w3(x))