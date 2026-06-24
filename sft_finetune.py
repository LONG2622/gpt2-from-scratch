#6.24实现SFT（supervised fine-tuning），将预训练语言模型调整为遵循人类指令，
# 使模型能够理解并执行人类指令。
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 新增：解决 SSL 证书问题
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import json
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from functools import partial
import matplotlib.pyplot as plt

#先导入LoRA模块
from lora import LoRAGPTModel

# ========== 新增：预训练权重加载函数 ==========
def load_pretrained_weights(model, config):
    """
    将 HuggingFace GPT-2 预训练权重加载到自定义 LoRA-GPT 模型
    优先尝试本地文件，失败则尝试从 HuggingFace 下载
    """
    print("\n正在加载 GPT-2 预训练权重...")
    
    from transformers import GPT2LMHeadModel
    
    # 优先尝试本地文件
    local_path = "./gpt2_pretrained"
    if os.path.exists(local_path) and os.path.isdir(local_path):
        print(f"尝试从本地加载: {local_path}")
        try:
            hf_model = GPT2LMHeadModel.from_pretrained(local_path)
            print("本地加载成功！")
        except Exception as e:
            print(f"本地加载失败: {e}")
            hf_model = None
    else:
        hf_model = None
    
    # 如果本地加载失败，尝试从 HuggingFace 下载
    if hf_model is None:
        print("尝试从 HuggingFace 下载...")
        try:
            hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
            print("从 HuggingFace 下载成功！")
        except Exception as e:  
            print(f"从 HuggingFace 下载失败: {e}")
            print("无法加载预训练权重，模型将使用随机初始化！")
            return model
    
    #1. 加载 Embedding 层
    model.tok_emb.weight.data = hf_model.transformer.wte.weight.data.clone()
    model.pos_emb.weight.data = hf_model.transformer.wpe.weight.data.clone()
    
    #2.加载 Transformer Block
    for i in range(config["n_layers"]):
        custom_block = model.trf_blocks[i]
        hf_block = hf_model.transformer.h[i] 
        # GPT-2 的 c_attn 包含 Q/K/V 权重（合并在一起）
        # GPT-2 c_attn 形状: (emb_dim, 3*emb_dim) = (768, 2304)
        qkv_weight = hf_block.attn.c_attn.weight.data
        qkv_bias = hf_block.attn.c_attn.bias.data
        
        d = config["emb_dim"]
        
        # 打印调试信息
        if i == 0:
            print(f"  Layer {i}: qkv_weight={qkv_weight.shape}, qkv_bias={qkv_bias.shape}")
            print(f"  Layer {i}: c_fc weight={hf_block.mlp.c_fc.weight.shape}, c_proj weight={hf_block.mlp.c_proj.weight.shape}")
        
        # GPT-2 的 Conv1D 权重是 (in, out)，需要转置为 (out, in) 给 nn.Linear
        # GPT-2 权重是 (emb_dim, 3*emb_dim)，按列分割
        # Q = [:, :d], K = [:, d:2d], V = [:, 2d:]
        qkv_weight_T = qkv_weight.t()  # 转置为 (3*emb_dim, emb_dim) = (2304, 768)
        
        q_weight = qkv_weight_T[:d, :].clone()     # (768, 768)
        k_weight = qkv_weight_T[d:2*d, :].clone()  # (768, 768)
        v_weight = qkv_weight_T[2*d:, :].clone()   # (768, 768)
        
        custom_block.att.W_query.weight.data = q_weight
        custom_block.att.W_key.weight.data = k_weight
        custom_block.att.W_value.weight.data = v_weight
        
        # 加载 Q/K/V bias
        custom_block.att.W_query.bias.data = qkv_bias[:d].clone()
        custom_block.att.W_key.bias.data = qkv_bias[d:2*d].clone()
        custom_block.att.W_value.bias.data = qkv_bias[2*d:].clone()
        
        # 输出投影
        custom_block.att.out_proj.weight.data = hf_block.attn.c_proj.weight.data.clone()
        custom_block.att.out_proj.bias.data = hf_block.attn.c_proj.bias.data.clone()
        
        # 加载 LayerNorm
        custom_block.norm1.scale.data = hf_block.ln_1.weight.data.clone()
        custom_block.norm1.shift.data = hf_block.ln_1.bias.data.clone()
        custom_block.norm2.scale.data = hf_block.ln_2.weight.data.clone()
        custom_block.norm2.shift.data = hf_block.ln_2.bias.data.clone()
        
        # 加载 FeedForward（GPT-2 使用 Conv1D，权重需要转置）
        # Conv1D 权重形状是 (in_features, out_features)，Linear 是 (out_features, in_features)
        custom_block.ff.layers[0].weight.data = hf_block.mlp.c_fc.weight.data.clone().t()  # 转置 (3072, 768)
        custom_block.ff.layers[0].bias.data = hf_block.mlp.c_fc.bias.data.clone()
        custom_block.ff.layers[2].weight.data = hf_block.mlp.c_proj.weight.data.clone().t()  # 转置 (768, 3072)
        custom_block.ff.layers[2].bias.data = hf_block.mlp.c_proj.bias.data.clone()
    
    #3. 加载最终层归一化和输出头
    model.final_norm.scale.data = hf_model.transformer.ln_f.weight.data.clone()
    model.final_norm.shift.data = hf_model.transformer.ln_f.bias.data.clone()
    model.out_head.weight.data = hf_model.lm_head.weight.data.clone()
    
    print("预训练权重加载成功！")
    return model
#配置
GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}
#准备数据集
def download_instruction_data():
    import urllib.request
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    file_path = "instruction-data.json"
    
    if not os.path.exists(file_path):
        try:
            print(f"正在下载数据集...")
            urllib.request.urlretrieve(url, file_path)
            print(f"数据集已下载: {file_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("使用本地示例数据...")
            # 创建示例数据（增加数量确保划分后各集都有数据）
            sample_data = [
                {"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进其性能，而无需进行明确编程。"},
                {"instruction": "翻译以下句子", "input": "Hello, world!", "output": "你好，世界！"},
                {"instruction": "写一首关于春天的诗", "input": "", "output": "春风拂柳绿丝绦，花开满园香气飘。燕子归来筑新巢，细雨绵绵润青苗。"},
                {"instruction": "解释什么是深度学习", "input": "", "output": "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次表示。"},
                {"instruction": "计算 2 + 2", "input": "", "output": "4"},
                {"instruction": "什么是GPT?", "input": "", "output": "GPT是Generative Pre-trained Transformer的缩写，是一种基于Transformer架构的大型语言模型。"},
                {"instruction": "解释什么是云计算", "input": "", "output": "云计算是通过互联网提供计算资源（如服务器、存储、数据库等）的一种服务模式。"},
                {"instruction": "翻译为英文", "input": "我爱学习", "output": "I love learning."},
                {"instruction": "解释什么是人工智能", "input": "", "output": "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。"},
                {"instruction": "写一个简短的故事", "input": "", "output": "小明在公园里散步，看到一只可爱的小猫正在追逐蝴蝶。阳光透过树叶洒下来，一切都显得那么美好。"},
                {"instruction": "解释什么是神经网络", "input": "", "output": "神经网络是一种模仿人脑神经元结构的计算模型，由多层节点组成，用于模式识别和预测。"},
                {"instruction": "翻译为中文", "input": "Machine learning is fun!", "output": "机器学习很有趣！"},
                {"instruction": "推荐一本好书", "input": "", "output": "《人类简史》是一本非常棒的书，它讲述了人类从原始社会到现代文明的发展历程。"},
                {"instruction": "解释什么是自然语言处理", "input": "", "output": "自然语言处理是人工智能的一个领域，研究如何让计算机理解和处理人类语言。"},
                {"instruction": "写一句励志名言", "input": "", "output": "成功不是将来才有的，而是从决定去做的那一刻起，持续累积而成。"},
                {"instruction": "解释什么是强化学习", "input": "", "output": "强化学习是一种机器学习方法，通过智能体与环境交互获得奖励来学习最优策略。"}
            ]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            print(f"已创建示例数据: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
#加载数据集
data = download_instruction_data()
print(len(data))
print(f"First example: \n{json.dumps(data[0], ensure_ascii=False, indent=2)}")
#提示词优化
def format_input(entry):
    """
    将数据条目格式化为统一的提示词格式
    
    输入: {"instruction": "...", "input": "...", "output": "..."}
    输出: "### Instruction: ...\n### Input: ...\n### Response: ..."
    """
    instruction_text = (
        f"Below is a instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction: {entry['instruction']}\n"
    )
    input_text = f"\n\n### Input: {entry['input']}\n" if entry["input"] else " "
    response_text = f"\n### Response: {entry['output']}"
    return instruction_text + input_text + response_text
#测试格式化
print("\nFormatted example:")
print(format_input(data[0])[:200]+"...")

#指令数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_texts = []

        for entry in data:
            full_text = format_input(entry)
            encoded = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
            if len(encoded) > max_length:
                encoded= encoded[:max_length]
            self.encoded_texts.append(encoded)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

#自定义批次处理函数
def custom_collate_fn(
        batch,
        pad_token_id= 50256,
        ignore_index = -100,
        device = "cpu"
):
    """
    将变长序列填充为等长批次，并构建 targets
    
    关键技巧:
    - inputs: 原始序列（去掉最后一个token）
    - targets: 偏移一个位置的序列（去掉第一个token）
    - 对于 padding 部分，使用 ignore_index 屏蔽损失
    """
    # 找到批次中最长的序列（加1是因为要做偏移）
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []
    targets_lst = []
    for item in batch:
        #添加结束符
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        #构建input 和 target 
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    return (
        torch.stack(inputs_lst).to(device),
        torch.stack(targets_lst).to(device)
    )
#划分数据集（确保各集至少有2个样本）
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_size = max(int(len(data) * train_ratio), len(data) - 4)  # 至少留4个给验证和测试
val_size = max(int(len(data) * val_ratio), 2)                 # 验证集至少2个
test_size = len(data) - train_size - val_size                 # 剩下的给测试

# 确保测试集至少有2个
if test_size < 2:
    test_size = 2
    val_size = max(val_size - (2 - test_size), 2)
    train_size = len(data) - val_size - test_size

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

print(f"Train size: {len(train_data)}")
print(f"Val size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

#创建数据加载器
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#自定义批次函数
customized_collate = partial(custom_collate_fn, 
                            device = device)
#创建数据集和加载器
batch_size = 2
max_length = 512
train_dataset = InstructionDataset(train_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset,
                           batch_size=batch_size,
                           collate_fn=customized_collate,
                           shuffle=True,
                           drop_last = True)
val_dataset = InstructionDataset(val_data, tokenizer, max_length)
val_loader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           collate_fn=customized_collate,
                           shuffle=False)

#创建模型
cfg = GPT_CONFIG_124M.copy()
cfg["context_length"] = max_length

#使用LoRA模型进行SFT
model = LoRAGPTModel(cfg,
                  lora_rank = 8,
                  lora_alpha = 8,
                  lora_dropout =0.0)

# ========== 新增：加载预训练权重 ==========
model = load_pretrained_weights(model, cfg)

model.to(device)

#查看参数统计
total_params = sum(p.numel() for p in model.parameters())
lora_params = model.count_lora_parameters()
print(f"\nTotal params: {total_params}")
print(f"Lora params: {lora_params}")
print(f"可训练参数比例:{lora_params/total_params * 100:.2f}%")  
#定义优化器和损失函数
optimizer = torch.optim.AdamW(
    model.get_lora_parameters(), 
    lr = 5e-4,
    weight_decay=0.01)
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0,1)
    targets_flat = target_batch.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat,
                                             ignore_index = -100)
    return loss
#训练函数
def train_sft(model, train_loader, val_loader, optimizer, device, num_epochs = 20):
    #在此调整训练参数，如学习率、批次大小等
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1

            if num_batches % 5 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

#验证（处理空验证集情况）
        model.eval()
        epoch_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                epoch_val_loss += loss.item()
                num_batches += 1
        if num_batches > 0:
            avg_val_loss = epoch_val_loss / num_batches
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = float('nan')
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: N/A (empty validation set)")
    return train_losses, val_losses

#开始训练
print("开始训练...")
train_losses, val_losses = train_sft(
    model, train_loader, val_loader, optimizer, device, num_epochs = 3)
#绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training and Validation Loss")
plt.show()

#测试生成
def format_input(entry):
    instruction_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}"
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def generate_response(model, tokenizer, instruction, input_text="",
                      max_new_tokens=128, temperature=0.7, top_p=0.9):
    model.eval()
    
    # 使用与训练时相同的格式
    prompt = format_input({"instruction": instruction, "input": input_text}) + "\n\n### Response:\n"
    
    #编码并生成
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # 温度采样
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Top-p 采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 移除概率超过 top_p 的 token
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            #遇到结束符停止
            if idx_next.item() == 50256:
                break
    
    #解码并提取响应
    full_text = tokenizer.decode(idx.squeeze(0).tolist())
    response = full_text[len(prompt):].strip()
    
    #清理响应
    response = response.split("###")[0].strip()
    return response
#测试几个例子
test_instructions = [
    {"instruction": "解释是什么是机器学习", "input": ""},
    {"instruction": "翻译一下句子", "input": "Hello, world!"}
]
print("\n" +"=" * 50)
print("测试生成效果")
print("="* 50)

for item in test_instructions:
    response = generate_response(model, tokenizer, item["instruction"],
                                  item["input"])
    print(f"Instruction: {item['instruction']}")
    if item["input"]:
        print(f"Input: {item['input']}")
    print(f"Generated Response: {response}")
    print("="*50)
#保存模型
model_path = "gpt_sft_lora_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已保存到 {model_path}")