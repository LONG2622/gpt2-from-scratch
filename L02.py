#编码注意力机制（encoding attention mechanisim）
import os
import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
input_query = inputs[1]
input_query
input_1 = inputs[0]
input_1

#做点积
torch.dot(input_query ,input_1)

torch.empty(inputs.shape[0])
query = inputs[1]

attn_score_2 = torch.empty(inputs.shape[0])
for i , x_i in enumerate(inputs):
    attn_score_2[i]= torch.dot (x_i, input_query)
print(attn_score_2)

#归一化计算注意力权重
attn_weight_2_tmp = attn_score_2 / attn_score_2.sum()
print("attention weights:",attn_weight_2_tmp)
print("sum:",attn_weight_2_tmp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)
attn_weight_2_naive = softmax_naive(attn_score_2)
print("attention weight:" , attn_weight_2_naive)
print("sum:" , attn_weight_2_naive.sum())

attn_weight_2 = torch.softmax(attn_score_2, dim = 0)


query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for x, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i] * x_i
print(context_vec_2)

for i,x_i in enumerate(inputs):
    print(f"{attn_weight_2[i]} --->{inputs[i]}")
    context_vec_2 += attn_weight_2[i] * x_i

#无权重的自注意力机制
attn_scores = torch.empty(6 ,6)

for i, x_i in enumerate (inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i , j] = torch.dot(x_i , x_j)
print(attn_scores)

attn_weight = torch.softmax(attn_scores , dim = 1)
attn_weight
attn_weight.sum(dim = -1)
torch.sum(torch.tensor([0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]))

attn_scores = inputs @ inputs.T
attn_weight = torch.softmax(attn_scores, dim = 1)
print(attn_weight)

all_context_vecs = attn_weight @ inputs
all_context_vecs

#计算所有token的权重
#先找到张量
inputs
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
#生成权重参数
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in , d_out))
W_key = torch.nn.Parameter(torch.rand(d_in ,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in ,d_out))
W_query
query_2 = x_2 @ W_query
query_2 

key = inputs @ W_key
value = inputs @ W_value

key_2 = key[1]
attn_score_22 = torch.dot(query_2 ,key_2)
attn_score_22

import math
attn_score_2 = query_2 @ key.T
attn_score_2
d_k = key.shape[1]
attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
attn_weights

attn_weights.sum()
context_vec_2 = attn_weights @ value 
#自注意力类
import os
import torch
import torch.nn as nn
m = torch.nn.Linear(2,3)
m.bias
class SelfAttention_V1(nn.Module):
    def __init__(self , d_in ,d_out):
        super().__init__()
        W_query = torch.nn.Parameter(torch.rand(d_in , d_out))
        W_key = torch.nn.Parameter(torch.rand(d_in ,d_out))
        W_value = torch.nn.Parameter(torch.rand(d_in ,d_out))
    def forward(self , x):
        queries = x @ W_query
        keys = x @ W_key
        values = x @ W_value
        attn_score_2 = query_2 @ keys.T
        attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @value
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in ,d_out)
sa_v1(inputs)
#添加了QKV偏差
m = torch.nn.Linear(2,3)
m.bias
class SelfAttention_V1(nn.Module):
    def __init__(self , d_in , d_out ,qkv_bias = False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_key = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_value = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
    def forward(self , x):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)
        attn_score_2 = query_2 @ key.T
        attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @value
        return context_vec

torch.manual_seed(123)
sa_v2 = SelfAttention_V1(d_in ,d_out)
sa_v2(inputs)

query = sa_v2.W_query(inputs)
key = sa_v2.W_key(inputs)
value = sa_v2.W_value(inputs)
attn_score_2 = query @ key.T
attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
attn_weights

context_length= attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length , context_length))
print(mask_simple)

mask_simple = attn_weights * mask_simple
mask_simple

mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool() , -torch.inf)
print(masked)

#用dropout掩码额外的注意力权重
torch.manual_seed(123)
layer = torch.nn.Dropout(0.5)
example= torch.ones(6,6)
example
layer(example)
dropout_rate = 0.5
1/(1- dropout_rate)

#实现因果注意力类
batch = torch.stack((inputs,inputs), dim =0)
batch.shape
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), 
                       diagonal = 1)
                       )

    def forward(self, x):
        b , num_tokens,d_in =x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores =queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens] , -torch.inf)
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
torch.manual_seed(789)
context_length = batch.shape[1]
dropout = 0.0
ca = CausalAttention(d_in = d_in, d_out = d_out,
                      context_length = context_length, dropout = dropout,
                    )
ca(batch)
"""
import torch
import torch.nn as nn

# 先把之前的变量准备好（你必须有这两个）
d_in = 256    # 输入维度
d_out = 256  # 输出维度

# 正确的因果注意力类
class CausalAttention(nn.Module):
    #修复1：__init__ 双下划线
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()  #修复2：正确写法
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # 修复3：正确的上三角掩码
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # 修复4：正确掩码
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        d_k = d_out
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

# 测试运行
torch.manual_seed(789)
context_length = 4  # 你的输入长度
dropout = 0.0

#修复5：参数匹配
ca = CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=dropout)

# 构造测试数据
batch = torch.randn(2, context_length, d_in)
output = ca(batch)
print("输出形状:", output.shape)"""

class MultiHeadAttentionWrapper(nn.Module):
    def  __init__ (self, d_in, d_out, context_length, dropout, num_heads=2, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
            ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
torch.manual_seed(123)
context_length=batch.shape[1]
d_in,d_out = batch.shape[0] ,2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
mha(batch)

#5.31
#实现多头注意力
import os
import torch
import torch.nn as nn
batch = torch.randn(2, 4, 256)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in,d_out, context_length, dropout , num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads ==0),\
        "d_out must be diveisible by num_heads"
        self.d_out = d_out
        self.num_heads= num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in , d_out , bias = qkv_bias)
        self.W_key = nn.Linear(d_in ,d_out ,bias = qkv_bias)
        self.W_value = nn.Linear(d_in , d_out , bias = qkv_bias)
        self.out_proj = nn.Linear(d_out , d_out )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal = 1)
        )
    def forward(self,x):
        b,num_tokens, d_in =x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens , self.num_heads, self.head_dim)
        values = values.view(b, num_tokens , self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens , self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2 )
        values = values.transpose(1, 2 )
        queries = queries.transpose(1, 2 )
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores /keys.shape[-1]**0.5 , dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b , num_tokens , self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
torch.manual_seed(123)
batch_size, context_length , d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in ,d_out ,context_length, num_heads = 2 ,dropout = 0.0)
context_vecs = mha(batch)
print(context_vecs)
print(context_vecs.shape)

#implementing a GPTmodel from scratch to  generate text
#训练模型进行文本生成5.31
#虚拟类

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn .Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn .Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn .Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"] , cfg["vocab_size"] , 
                                 bias =False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds= self.pos_emb(
            torch.arange(seq_len , device = in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x =self.drop_emb(x)
        x =self.trf_blocks(x)
        x =self.final_norm(x)
        logits = self.out_head(x)
        return logits
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    """
    残差连接：
    def forward (self,x):
        attn_out =self.attn(self.norm1(x))
        x = x + self.drop(att_out)
    """
    def forward(self , x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
    def forward(self, x):
        return x
    
#分词器
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim" :768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias" : False
}
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch , dim =0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print(logits.shape)
print(logits)
#实现层归一化layer——normalization(防止梯度消失和梯度爆炸)
torch.manual_seed(123)
batch_example = torch.randn(2,5)
batch_example
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
out.mean()#均值
out.var()#方差
var = out.var(dim= -1, keepdim = True)
var
mean = out.mean(dim=-1, keepdim=True)  # 先算出 out 的均值！
var = out.var(dim=-1, keepdim=True)    # 算出方差

out_norm = (out - mean) / torch.sqrt(var + 1e-5)  # 这才是层归一化
(out-mean).mean(dim = -1, keepdim = True)
torch.set_printoptions(sci_mode= False)

#层归一化类
class Layernorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5#(防止分母0出现)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self , x):
        mean = x.mean(dim =-1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased =False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

ln = Layernorm(emb_dim = 5)
out_ln= ln(batch_example)
mean = out_ln.mean(dim = -1, keepdim = True)
var = out_ln.var(dim = -1 ,unbiased = False , keepdim = True)
print(mean)
print(var)

#GELU激活函数的实现
class GELU(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class RELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x , 3))
        ))
    #可视化绘制图像比较ReLU 和GELU函数
import matplotlib.pyplot as plt
gelu , relu = nn.GELU() , nn.ReLU()
x = torch.linspace(-3, 3 , 100)
y_gelu ,y_relu = gelu(x) , relu(x)
plt.figure(figsize= (8 ,3))
for i ,(y , label) in enumerate (zip([y_gelu , y_relu], ["GELU", "RELU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x ,y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

#前馈神经网络模块类的实现
#768 -> 768 * 4
class FeedForward(nn.Module):
    def __init__(self ,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                      nn.GELU(),
                      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                      )
    def forward(self,x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2 , 3 , 768)
out = ffn(x)
print(out.shape)


#添加快捷连接（神经网络的实现）
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__ (self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0] , layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1] , layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2] , layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3] , layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4] , layer_sizes[5]), GELU()),
        ])    
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x 
#反向传播过程中计算梯度函数：
def print_gradients(model , x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output , target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1. , 0.,-1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes , use_shortcut = False
)

print_gradients(model_without_shortcut, sample_input)

#包含跳跃连接的模型：
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut= True
)
print_gradients(model_with_shortcut , sample_input)

#transform block(实现其中的注意力层和线性层)
#GPT 的transformer 块组件
class TransformerBlock(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
        self.att =MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias =cfg["qkv_bias"]
        ) 
        self.ff = FeedForward(cfg)
        self.norm1 = Layernorm(cfg["emb_dim"])
        self.norm2 = Layernorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        #注意力残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
#前馈网络残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x 
#测试
torch.manual_seed(123)
x = torch.rand(2 , 4 ,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
x.shape
#实现GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn .Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn .Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn .Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = Layernorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"] , cfg["vocab_size"] , 
                                 bias =False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds= self.pos_emb(
            torch.arange(seq_len , device = in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x =self.drop_emb(x)
        x =self.trf_blocks(x)
        x =self.final_norm(x)
        logits = self.out_head(x)
        return logits

#测试
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print(batch)
print(out )
print(out.shape)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    """
    残差连接：
    def forward (self,x):
        attn_out =self.attn(self.norm1(x))
        x = x + self.drop(att_out)
    """
    def forward(self , x):
        return x
    
class LayerNorm(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
    def forward(self, x):
        return x

#numel求模型参数张量的总参数量
total_param = sum(p.numel() for p in model.parameters())
print(total_param)

# 看各部分参数
def count_params(module):
    return sum(p.numel() for p in module.parameters())

#共享权重得到1.24亿参数
total_params_gpt2 = (
    total_param - sum(p.numel()
                      for p in model.out_head.parameters())
)
print(total_params_gpt2)

#实现文本生成
#softmax 并不必要
def generate_text_simple(model , idx, max_new_tokens , context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():#反向训练算法
            logits = model(idx_cond)
        logits = logits[:, -1 , :]
        probas = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next) , dim = 1)
    return idx
torch.argmax(torch.tensor([14, 1 , -2 ,1]))#查找索引位置的函数(找最大值)
start_context = "Hello , I am" 
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor.shape)
out = generate_text_simple(
    model = model ,
    idx = encoded_tensor, 
    max_new_tokens = 6 ,
    context_size = GPT_CONFIG_124M["context_length"])
#准备训练模型
print(out)
print(len(out[0]))
decode_text =  tokenizer.decode(out.squeeze(0).tolist())
print(decode_text)
#6.1
#开始预训练
import torch
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim" :768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias" : False
}
torch.manual_seed(123)
model = (GPTModel(GPT_CONFIG_124M))
model.eval()

import tiktoken
def text_to_token_ids(text , tokenizer):
    encoded = tokenizer.encode(text , allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer) :
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer) ,
    max_new_tokens=10, 
    context_size=GPT_CONFIG_124M["context_length"]
)

print(token_ids_to_text(token_ids , tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],
                       [40,  1107,588   ]])
#every effot moves
#i really like
targets= torch.tensor([[3626, 6100 , 345],
                       [1107, 288, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas =torch.softmax(logits, dim = -1)
print(probas.shape)

#计算文本生成损失(交叉熵损失 )

token_ids = torch.argmax(probas, dim = -1, keepdim = True)
print(token_ids)
print(token_ids_to_text(targets[0], tokenizer))
print(token_ids_to_text(token_ids[0].flatten(), tokenizer))

text_idx =0
target_probas_1 = probas[text_idx,[0, 1, 2],targets[text_idx]]
print(target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx,[0, 1, 2],targets[text_idx]]
print(target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

#交叉熵损失函数cross_entropy
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print(logits_flat.shape)
print(targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat , targets_flat)
print(loss)
#计算训练集和测试集的损失
file_path = r"C:\Users\田建隆\Downloads\LLMs-from-scratch-main\the-verdict.txt"
with open(file_path, "r", encoding = "utf-8") as file:
    text_data= file.read()
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(total_characters)
print(total_tokens)


 #完整dataloader函数

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256,
                         stride=128, drop_last=True, shuffle=True, num_workers=0):

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader
    #完整dataloader函数
def create_dataloader_v1(txt , batch= 4 ,max_length = 256,stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last ,
        num_workers = num_workers
    )
    return dataloader
#使用90%的数据训练
train_ratio= 0.90
split_idx = int(train_ratio* len(text_data))
train_data = text_data[:split_idx]
val_data =text_data[split_idx:]
#创建相应的数据加载器
train_loader = create_dataloader_v1(
    train_data,
    tokenizer = tokenizer,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last= True,
    shuffle= True,
    num_workers = 0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    tokenizer = tokenizer,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = False,
    shuffle = False,
    num_workers = 0
)
for x, y in train_loader:
    print(x.shape, y.shape)

for x, y in val_loader:
    print(x.shape, y.shape)
#工具函数，用于计算返回给定批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    ) 
    return loss

def calc_loss_loader(data_loader , model , device, num_batches = None):
    total_loss= 0.
    if len(data_loader) ==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i , (input_batch, target_batch) in  enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model , device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

device = torch.device("cpu")
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model ,device)
    val_loss= calc_loss_loader(val_loader, model, device)
print(train_loss)
print(val_loss)
#开始正式训练模型：预训练大模型的主函数
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen , global_step= 0 , -1
    for epoch in  range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch , model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print({epoch+1}, {train_loss},{val_loss})
    generate_and_print_sample(
        model , tokenizer, device, start_context            )
    return train_losses, val_losses, track_tokens_seen 
def evaluate_model(model ,train_loader , val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model , device, num_batches= eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model ,device, num_batches= eval_iter 
        )
        model.train()
        return train_loss, val_loss
#便捷函数，用于跟踪模型在训练过程中是否有所改进    
def generate(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
def generate_and_print_sample(model , tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model  =model, idx = encoded,
            max_new_tokens = 50, context_size = context_size
        )
    decoded_text = token_ids_to_text (token_ids, tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()

#用Adaw优化器进行十轮训练
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0004, weight_decay = 0.1
)
num_epochs = 10
train_losses, val_losses, token_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq= 5, eval_iter = 5,
    start_context="Every effort moves you", tokenizer= tokenizer
)
#豆包
# --------------- 1. 切分数据（完全正确版）---------------
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]  # 干净、正确、无任何空格错误

# --------------- 2. 重建数据加载器 ---------------
train_loader = create_dataloader_v1(
    train_data,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# --------------- 3. 重置模型 + 重新训练 ---------------
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 15

train_losses, val_losses, token_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

#6.2训练集和测试集的损失可视化
import matplotlib.pyplot as plt
from matplotlib.ticker import MAxNLocator
def plot_losses(epochs_seen , tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize = (5,3))
    ax1.plot(epochs_seen, train_losses, label = "Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle= "-.", label = "Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MAxNLocator(integer = True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen,train_losses, alpha = 0 )
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

#6.2控制随机性的解码策略
model.to("cpu")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model = model, idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("output the text:\n",
      token_ids_to_text(token_ids, tokenizer))
#温度缩放temperature scaling
vocab = {
    "closer":0,
    "every":1,
    "effort":2,
    "forward":3,
    "inches":4,
    "moves":5,
    "pizza":6,
    "toward":7,
    "you":8,
}
inverse_vocab = {v: k for k ,v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51,0.89,-1.90,6.75,1.63,-1.62,-1.89,6.28,1.79] 
)
probas = torch.softmax(next_token_logits,dim= 0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples = 1).item()
print(inverse_vocab[next_token_id])
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples = 1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
#温度缩放
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim = 0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize = (5,3))
for  i, T in enumerate(temperatures):
    rects = ax.bar(x+ i * bar_width , scaled_probas[i],
                   bar_width, label = f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation = 90)
ax.legend()
plt.tight_layout()
plt.show()
#top-k采样
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print(top_logits)
print(top_pos)