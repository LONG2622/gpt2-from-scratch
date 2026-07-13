#ch03 -> gpt_model.py
#implementing a GPTmodel from scratch to  generate text 
#训练模型进行文本生成5.31
#分词器
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch 
import torch.nn as nn
import tiktoken
import math
from attention import *
from data_preprocessing import *
#GPT模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim" :768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias" : False
}
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
mean = out.mean(dim=-1, keepdim=True)  # 先算出 out 的均值
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
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
        (x + 0.044715 * torch.pow(x, 3))))
class RELU(nn.Module):
    def forward(self, x):
        return torch.max(torch.tensor(0.0), x)
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
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes , use_shortcut = False
)
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

print_gradients(model_without_shortcut, sample_input)
#包含跳跃连接的模型：
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut= True
)
print_gradients(model_with_shortcut , sample_input)
#transform block(实现其中的注意力层和线性层)
#GPT 的transformer 块组件
from attention import MultiHeadAttention
class TransformerBlock(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
        self.att =MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias =cfg["qkv_bias"]) 
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
#实现GPT模型架构
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
    
#numel求模型参数张量的总参数量
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

# 看各部分参数
def count_params(module):
    return sum(p.numel() for p in module.parameters())
#共享权重得到1.24亿参数
total_params_gpt2 = (
    total_params - sum(p.numel()
                      for p in model.out_head.parameters())
)
print(total_params_gpt2)
#计算GPTModel对象中所有参数的内存需求
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 *1024)
print(f"total size of the model: {total_size_mb:.2f}")
#开始生成文本
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

model.eval()
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