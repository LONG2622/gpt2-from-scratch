from ch01 import *
#编码注意力机制（encoding attention mechanisim）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
#直接用pytorch的softmax
attn_weight_2 = torch.softmax(attn_score_2, dim = 0)

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for x, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i] * x_i
print(context_vec_2)
#无权重的自注意力机制
attn_scores = torch.empty(6 ,6)

for i, x_i in enumerate (inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i , j] = torch.dot(x_i , x_j)
attn_scores = inputs @ inputs.T#也可直接使用矩阵乘法
print(attn_scores)

#归一化操作
attn_weight = torch.softmax(attn_scores , dim = 1)
attn_weight
attn_weight.sum(dim = -1)
torch.sum(torch.tensor([0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]))

attn_scores = inputs @ inputs.T
attn_weight = torch.softmax(attn_scores, dim = 1)
print(attn_weight)

all_context_vecs = attn_weight @ inputs
all_context_vecs
#实现带可训练权重的自注意力机制
#先逐步计算注意力权重
#计算所有token的权重
#先找到张量
inputs
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
#生成权重参数
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in , d_out),requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in ,d_out),requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in ,d_out),requires_grad = False)
W_query
query_2 = x_2 @ W_query
query_2 
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = x_2 @ W_key
values = x_2 @ W_value
"""
keys_2 = keys[1]
attn_score_22 = torch.matmul(query_2.unsqueeze(0), keys_2.unsqueeze(-1)).squeeze()
attn_score_22"""
import math
attn_score_2 = query_2 @ keys.T
attn_score_2
d_k = keys.shape[1]
attn_weights_2 =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
attn_weights_2

attn_weights_2.sum()
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
#实现一个简化的自注意力类
import os
import torch
import torch.nn as nn
class SelfAttention_V1(nn.Module):
    def __init__(self , d_in ,d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in , d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in ,d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in ,d_out))
    def forward(self , x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights =torch.softmax(
            attn_scores / keys.shape[-1]** 0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec
#
torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in ,d_out)
print(sa_v1(inputs))
#v2：使用了pytorch线性层的自注意力类
class SelfAttention_V2(nn.Module):
    def __init__(self , d_in , d_out ,qkv_bias = False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_key = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_value = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
    def forward(self , x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights =torch.softmax(
            attn_scores / keys.shape[-1]** 0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_V2(d_in ,d_out)
print(sa_v2(inputs))

#因果注意力掩码的实现
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)
attn_scores = queries @ keys.T
attn_weights =torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
attn_weights
#创建掩码
context_length= attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length , context_length))
print(mask_simple)
masked_simple = attn_weights * mask_simple
masked_simple
row_sums = masked_simple.sum(dim = -1, keepdim = True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool() , -torch.inf)
print(masked)
#对掩码结果使用softmax函数更高效
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim =1)
print(attn_weights)

#用dropout掩码额外的注意力权重
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example= torch.ones(6,6)
example
#对注意力权重做dropout操作
torch.manual_seed(123)
print(dropout(attn_weights))

#实现因果注意力类
batch = torch.stack((inputs,inputs), dim =0)
batch.shape
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, 
                 dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
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
        attn_weights = torch.softmax(attn_scores /keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
torch.manual_seed(123)
context_length = batch.shape[1]
dropout = 0.0
ca = CausalAttention(d_in = d_in, d_out = d_out,
                      context_length = context_length, dropout = dropout,
                    )
ca(batch)
ca(batch).shape
#实现多头注意力的封装类
class MultiHeadAttentionWrapper(nn.Module):
    def  __init__ (self, d_in, d_out, context_length,
                dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
            for _ in range(num_heads)
            ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(123)
context_length=batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
print(mha(batch))
#5.31
#将多头功能整合到一个类，实现高效多头注意力
import os
import torch
import torch.nn as nn
batch = torch.randn(2, 4, 256)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in,d_out, 
                 context_length, dropout , num_heads, qkv_bias = False):
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
        b, num_tokens, d_in =x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens , self.num_heads, self.head_dim)
        values = values.view(b, num_tokens , self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens , self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)
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
#简单用法
torch.manual_seed(123)
batch_size, context_length , d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in ,d_out ,context_length, num_heads = 2 ,dropout = 0.0)
context_vecs = mha(batch)
print(context_vecs)
print(context_vecs.shape)