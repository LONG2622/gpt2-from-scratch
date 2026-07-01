#ch06 -> instruction_finetune.py
# 通过微调遵循人类指令 6.5
# 先为有监督微调准备数据集
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import time
import torch
import tiktoken
from functools import partial
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# 使用统一的配置和工具函数
from src.configs import MODEL_CONFIGS, TrainingConfig, GenerationConfig
from src.utils import generate, download_and_load_file, save_json, save_model
from src.training import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    text_to_token_ids,
    token_ids_to_text,
)

# 导入模型相关模块
from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel

# 打印依赖版本
from importlib.metadata import version
pkgs = ["numpy", "matplotlib", "tiktoken", "torch", "tqdm"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 下载数据集
file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
data = download_and_load_file(file_path, url)
print(f"数据总量: {len(data)}")

# 提示词格式化
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# 划分数据集
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

# 指令数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# 自定义批次函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return (
        torch.stack(inputs_lst).to(device),
        torch.stack(targets_lst).to(device)
    )

# 初始化 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 加载数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

batch_size = 4
num_workers = 0

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=True, drop_last=True, num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)

# 加载 GPT 模型
CHOOSE_MODEL = "gpt2-small"
BASE_CONFIG = MODEL_CONFIGS[CHOOSE_MODEL]

model = GPTModel(BASE_CONFIG.__dict__)
model.eval()
model.to(device)

# 测试生成
torch.manual_seed(123)
input_text = format_input(val_data[0])
print("\n输入提示词:\n", input_text)

gen_config = GenerationConfig(
    max_new_tokens=256,
    eos_id=50256
)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=gen_config.max_new_tokens,
    context_size=BASE_CONFIG.context_length,
    eos_id=gen_config.eos_id
)

generated_text = token_ids_to_text(token_ids, tokenizer)
response = generated_text[len(input_text):].strip()
print("\n微调前模型回复:\n", response)

# 开始微调
model.train()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 1

# 训练模型
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

                # 生成示例
                model.eval()
                context_size = model.pos_emb.weight.shape[0]
                encoded = text_to_token_ids(start_context, tokenizer).to(device)
                with torch.no_grad():
                    gen_ids = generate(model=model, idx=encoded, max_new_tokens=20, context_size=context_size)
                decoded = token_ids_to_text(gen_ids, tokenizer)
                print(f"生成示例: {decoded.strip()}")
                model.train()

    return train_losses, val_losses, track_tokens_seen

start_time = time.time()
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="### Instruction:\nHello", tokenizer=tokenizer
)
end_time = time.time()
print(f"训练耗时: {(end_time - start_time)/60:.2f} 分钟")

# 画损失曲线
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 生成测试集回复
model.eval()
torch.manual_seed(123)

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=gen_config.max_new_tokens,
        context_size=BASE_CONFIG.context_length,
        eos_id=gen_config.eos_id
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    test_data[i]["model_response"] = response_text

# 保存结果
save_json(test_data, "instruction-data-with-response.json")

# 保存模型
file_name = f"{CHOOSE_MODEL}-sft.pth"
save_model(model, file_name)