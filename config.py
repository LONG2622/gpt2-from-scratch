#新建config 用于同一配置管理
from dataclasses import dataclass
@dataclass
class ModelConfig:
    vocab_size: int = 5057
    context_length: int = 1024
    emb_dim : int  = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias = False

@dataclass
class TraningConfig:
    batch_size : int = 8
    lr: float = 3e-4
    num_epochs :int = 10
    weight_decay: float = 0.1