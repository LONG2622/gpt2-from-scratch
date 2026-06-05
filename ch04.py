#在无标签数据上进行预训练
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ch03 import GPTModel, generate_text_simple
from ch03 import *
from ch02 import *
from ch01 import *
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
#从文本到tokenID转换的工具函数
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
#计算文本生成损失
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
#对概率分数应用对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
#计算平均对数概率分数
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

#交叉熵损失函数cross_entropy
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print(logits_flat.shape)
print(targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat , targets_flat)
print(loss)

#计算训练集和测试集的损失
file_path = "the-verdict.txt"
with open(file_path, "r", encoding = "utf-8") as file:
    text_data= file.read()
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(total_characters)
print(total_tokens)

#90%数据作为训练集，剩余为测试集
train_ratio= 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
import torch
from torch.utils.data import Dataset, DataLoader
from ch01 import create_dataloader_v1 , GPTDatasetV1
torch.manual_seed(123)

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
#输出
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
#用于计算训练集和测试集损失的函数

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
#实际应用
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
#定义文本生成函数
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

#6.2训练集和测试集的损失可视化
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize = (5,3))
    ax1.plot(epochs_seen, train_losses, label = "Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle= "-.", label = "Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
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

new_logits = torch.where(
    condition = next_token_logits< top_logits[-1],
    input = torch.tensor(float('-inf')),
    other = next_token_logits
)
print(new_logits)
#6.3修改文本生成函数modifying the text function
def generate(model, idx, max_new_tokens, context_size,
             temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits =torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            ) 
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples= 1)
        else:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
#测试
torch.manual_seed(123)
token_ids = generate(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k =25,
    temperature = 1.4)
print(token_ids_to_text(token_ids, tokenizer))
#使用pytorch加载和保存模型权重
torch.save(model.state_dict(), "model.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth",
                                 map_location = device))
model.eval()
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
},
"model_and_optimizer.pth"
)
#加载训练权重
"""import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05"
    "01_main-chapter-code/gpt_download.py"
)
filename= url.split('/')[-1]
urllib.request.urlretrieve(url, filename)
#下载7个相关文件

from gpt_download import download_and_load_gpt2
settings , params = download_and_load_gpt2(
    model_size = "124M", models_dir = "gpt2"
)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim" :768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias" : False
}
model_config = {
    "gpt2-small  (124M)":{"emb_dim":768, "n_layers":12, "n_heads":12},
    "gpt2-medium (355M)":{"emb_dim":1024, "n_layers":24, "n_heads":16},
    "gpt2-large  (774M)":{"emb_dim":1280, "n_layers":36, "n_heads":20},
    "gpt2-xl  (1558M)":{"emb_dim":1600, "n_layers":48, "n_heads":25},
}

NEW_CONFIG = GPT_CONFIG_124M.copy()
model_name = "124M"  
model_config = {"124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}}
NEW_CONFIG.update(model_config[model_name])
gpt = GPTModel(NEW_CONFIG)
gpt.eval()
#将tensorflow转为pytorch
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch.Left:{left.shape},"
                         "Right:{right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))
#将openai的权重加载到GPT模型代码中
import numpy as np
def load_weights_into_gpt(gpt,params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wpe'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis = -1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis= -1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
        
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

load_weights_into_gpt(gpt, params)
gpt.to(device)

#测试
torch.manual_seed(123)
token_ids = generate(
    model = gpt,
    idx = text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens= 25,
    context_size = NEW_CONFIG["context_length"],
    top_k =50,
    temperature= 1.5
    )
print(token_ids_to_text(token_ids, tokenizer))
"""