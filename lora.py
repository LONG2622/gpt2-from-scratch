# c:\Users\田建隆\Desktop\github-repository\build-a-LLM-from-scratch\lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== LoRA 核心层 ====================
class LoRALayer(nn.Module):
    """
    LoRA 层实现
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        rank: 低秩维度（通常取 8, 16, 32）
        alpha: 缩放因子，默认等于 rank
        dropout: dropout 率
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, 
                 alpha: float = None, dropout: float = 0.0):
        super().__init__()
        
        # 低秩矩阵 A (in_dim x rank)，初始化为很小的随机值
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        # 低秩矩阵 B (rank x out_dim)，初始化为零
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # 缩放因子，使用 alpha / rank 作为缩放因子（LoRA 论文推荐做法）
        self.scaling = (alpha / rank) if alpha is not None else 1.0
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:x @ A @ B * scaling
        
        Args:
            x: 输入张量，形状 (..., in_dim)
        
        Returns:
            输出张量，形状 (..., out_dim)
        """
        # 应用 dropout（可选）
        if self.dropout is not None:
            x = self.dropout(x)
        
        # 计算低秩投影
        result = (x @ self.A @ self.B) * self.scaling
        return result

#层归一化(从 ch03 复制)
class Layernorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# ==================== 前馈网络（从 ch03 复制） ====================    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)

# ==================== 支持 LoRA 的多头注意力 ====================
class LoRAMultiHeadAttention(nn.Module):
    """
    支持 LoRA 的多头注意力层
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 num_heads: int, dropout: float = 0.0, qkv_bias: bool = False,
                 lora_rank: int = 8, lora_alpha: float = None, 
                 lora_dropout: float = 0.0):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # 原始 Q/K/V 投影层（GPT-2 始终使用 bias）
        self.W_query = nn.Linear(d_in, d_out, bias=True)
        self.W_key = nn.Linear(d_in, d_out, bias=True)
        self.W_value = nn.Linear(d_in, d_out, bias=True)
        
        # LoRA 层（应用于 Q 和 V）
        self.lora_query = LoRALayer(d_in, d_out, lora_rank, lora_alpha, lora_dropout)
        self.lora_value = LoRALayer(d_in, d_out, lora_rank, lora_alpha, lora_dropout)
        
        # 输出投影层（GPT-2 c_proj 有 bias）
        self.out_proj = nn.Linear(d_out, d_out, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 因果掩码
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
        # 冻结原始权重
        self.freeze_original_weights()
    
    def freeze_original_weights(self):
        """冻结原始 Q/K/V 权重"""
        for param in self.W_query.parameters():
            param.requires_grad = False
        for param in self.W_key.parameters():
            param.requires_grad = False
        for param in self.W_value.parameters():
            param.requires_grad = False
        for param in self.out_proj.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        
        # 原始 Q/K/V 投影
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)        
        # 添加 LoRA 增量（始终应用）
        queries += self.lora_query(x)
        values += self.lora_value(x)
        
        # 重塑为多头格式
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 应用因果掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 归一化
        attn_weights = F.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec

#支持 LoRA 的 Transformer Block
class LoRATransformerBlock(nn.Module):
    """
    支持 LoRA 的 Transformer Block
    """
    def __init__(self, cfg: dict, lora_rank: int = 8, lora_alpha: float = None, 
                 lora_dropout: float = 0.0):
        super().__init__()
        
        # 支持 LoRA 的注意力层
        self.att = LoRAMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # 前馈网络（冻结）
        self.ff = FeedForward(cfg)
        for param in self.ff.parameters():
            param.requires_grad = False
        
        # 层归一化（冻结）
        self.norm1 = Layernorm(cfg["emb_dim"])
        for param in self.norm1.parameters():
            param.requires_grad = False
        
        self.norm2 = Layernorm(cfg["emb_dim"])
        for param in self.norm2.parameters():
            param.requires_grad = False
        
        # Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        # 前馈网络残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x

#支持 LoRA 的 GPT 模型
class LoRAGPTModel(nn.Module):
    """
    支持 LoRA 的 GPT 模型
    """
    def __init__(self, cfg: dict, lora_rank: int = 8, lora_alpha: float = None, 
                 lora_dropout: float = 0.0):
        super().__init__()
        # Token 嵌入（冻结）
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        for param in self.tok_emb.parameters():
            param.requires_grad = False
        
        # 位置嵌入（冻结）
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        for param in self.pos_emb.parameters():
            param.requires_grad = False
        
        # Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # LoRA Transformer Blocks
        self.trf_blocks = nn.Sequential(
            *[LoRATransformerBlock(cfg, lora_rank, lora_alpha, lora_dropout)
              for _ in range(cfg["n_layers"])]
        )
        
        # 最终层归一化（冻结）
        self.final_norm = Layernorm(cfg["emb_dim"])
        for param in self.final_norm.parameters():
            param.requires_grad = False
        
        # 输出头（冻结）
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        for param in self.out_head.parameters():
            param.requires_grad = False
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        
        # 获取嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # 相加并 dropout
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # 经过 Transformer Blocks
        x = self.trf_blocks(x)
        
        # 最终归一化和输出
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    def get_lora_parameters(self):
        """获取所有 LoRA 参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                if param.requires_grad:
                    lora_params.append(param)
        return lora_params
    
    def count_lora_parameters(self):
        """计算 LoRA 参数数量"""
        return sum(p.numel() for p in self.get_lora_parameters())

#工具函数
def convert_to_lora(model: nn.Module, lora_rank: int = 8, lora_alpha: float = None, 
                    lora_dropout: float = 0.0) -> nn.Module:
    """将普通 GPT 模型转换为支持 LoRA 的模型"""
    # 需要导入原始的 MultiHeadAttention
    try:
        from ch02 import MultiHeadAttention
    except ImportError:
        print("警告：无法导入 MultiHeadAttention，转换功能不可用")
        return model
    
    # 遍历所有模块
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # 创建新的 LoRA 注意力层
            new_attn = LoRAMultiHeadAttention(
                d_in=module.W_query.in_features,
                d_out=module.W_query.out_features,
                context_length=module.mask.shape[0],
                num_heads=module.num_heads,
                dropout=module.dropout.p,
                qkv_bias=module.W_query.bias is not None,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            
            # 复制原始权重
            new_attn.W_query.load_state_dict(module.W_query.state_dict())
            new_attn.W_key.load_state_dict(module.W_key.state_dict())
            new_attn.W_value.load_state_dict(module.W_value.state_dict())
            new_attn.out_proj.load_state_dict(module.out_proj.state_dict())
            
            # 替换模块
            setattr(parent, name.split('.')[-1], new_attn)
    return model
def freeze_all_except_lora(model: nn.Module):
    """冻结除 LoRA 参数外的所有参数"""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True