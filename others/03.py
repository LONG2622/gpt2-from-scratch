#因果注意力掩码
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
        query = inputs @ W_query
        key = inputs @ W_key
        value = inputs @ W_value
        attn_score_2 = query_2 @ key.T
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

