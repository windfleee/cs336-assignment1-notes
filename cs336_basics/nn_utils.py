import torch
import torch.nn as nn
from jaxtyping import Float, Int
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