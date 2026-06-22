# lr.py 修改导入
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 使用独立的配置文件
from config_clean import GPT_CONFIG_124M
from lora import LoRAGPTModel
# 在 lr.py 中添加以下代码
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0, tokenizer=None):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

# 划分数据集
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 创建数据加载器
train_loader = create_dataloader_v1(
    train_data,
    tokenizer=tokenizer,
    batch_size=4,
    max_length=128,
    stride=64,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    tokenizer=tokenizer,
    batch_size=4,
    max_length=128,
    stride=128,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

# 创建模型
cfg = GPT_CONFIG_124M.copy()
cfg["context_length"] = 128  # 使用较短的上下文进行演示

# 方式1：直接创建 LoRA 模型
model = LoRAGPTModel(
    cfg,
    lora_rank=8,
    lora_alpha=8,
    lora_dropout=0.0
)

# 方式2：将现有模型转换为 LoRA 模型
# model = GPTModel(cfg)
# model = convert_to_lora(model, lora_rank=8)
# freeze_all_except_lora(model)

model.to(device)

# 查看可训练参数数量
total_params = sum(p.numel() for p in model.parameters())
lora_params = model.count_lora_parameters()
print(f"总参数: {total_params:,}")
print(f"LoRA 参数: {lora_params:,}")
print(f"可训练比例: {lora_params / total_params * 100:.2f}%")

# 定义优化器（只优化 LoRA 参数）
optimizer = torch.optim.AdamW(
    model.get_lora_parameters(),
    lr=5e-4,
    weight_decay=0.01
)

# 定义损失函数
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()
    return nn.functional.cross_entropy(logits_flat, targets_flat)

# 训练函数
def train_lora(model, train_loader, val_loader, optimizer, device, num_epochs=5):
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print()
    
    return train_losses, val_losses

# 开始训练
print("=" * 50)
print("开始 LoRA 微调")
print("=" * 50)
train_losses, val_losses = train_lora(model, train_loader, val_loader, optimizer, device, num_epochs=5)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LoRA Fine-tuning Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# 测试文本生成
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    model.eval()
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx.squeeze(0).tolist())

# 生成示例
prompt = "Every effort moves you"
generated = generate_text(model, tokenizer, prompt, max_new_tokens=30)
print("生成文本:")
print(generated)

# 保存 LoRA 模型
torch.save(model.state_dict(), "gpt_lora_model.pth")
print("\nLoRA 模型已保存到 gpt_lora_model.pth")