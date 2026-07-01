#ch04 -> pretrain_trainer.py
# 在无标签数据上进行预训练
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 使用统一的配置和工具函数
from src.configs import MODEL_CONFIGS, TrainingConfig
from src.utils import generate, generate_text_simple
from src.training import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    text_to_token_ids,
    token_ids_to_text,
)

# 导入模型相关模块
from gpt_model import GPTModel
from data_preprocessing import create_dataloader_v1

# 使用统一配置
GPT_CONFIG_124M = MODEL_CONFIGS["gpt2-small"]
GPT_CONFIG_124M.context_length = 256  # 演示用较短的上下文

# 初始化模型
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M.__dict__)
model.eval()

# 初始化分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 测试文本生成
start_context = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M.context_length
)
print("初始生成文本:", token_ids_to_text(token_ids, tokenizer))

# 计算文本生成损失（演示）
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345], [1107, 288, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("示例损失:", loss.item())

# 加载数据集
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

print(f"文本字符数: {len(text_data)}")
print(f"Token数: {len(tokenizer.encode(text_data))}")

# 划分训练集和验证集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 创建数据加载器
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=GPT_CONFIG_124M.context_length,
    stride=GPT_CONFIG_124M.context_length,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    tokenizer=tokenizer,
    batch_size=2,
    max_length=GPT_CONFIG_124M.context_length,
    stride=GPT_CONFIG_124M.context_length,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 测试数据加载器
for x, y in train_loader:
    print("训练集batch形状:", x.shape, y.shape)
    break

for x, y in val_loader:
    print("验证集batch形状:", x.shape, y.shape)
    break

# 计算初始损失
device = torch.device("cpu")
model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"初始训练损失: {train_loss:.4f}")
print(f"初始验证损失: {val_loss:.4f}")

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
        token_ids = generate(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("训练后生成:", decoded_text.replace("\n", " "))
    model.train()

    return train_losses, val_losses, track_tokens_seen

# 使用AdamW优化器训练
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M.__dict__)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)

train_config = TrainingConfig(
    num_epochs=10,
    eval_freq=5,
    eval_iter=5
)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=train_config.num_epochs,
    eval_freq=train_config.eval_freq,
    eval_iter=train_config.eval_iter,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

# 绘制损失曲线
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

epochs_tensor = torch.linspace(0, train_config.num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 测试温度缩放和Top-K采样
model.to("cpu")
model.eval()

# 温度缩放演示
vocab = {"closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4,
         "moves": 5, "pizza": 6, "toward": 7, "you": 8}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
fig, ax = plt.subplots(figsize=(5, 3))
x = torch.arange(len(vocab))
bar_width = 0.15

for i, T in enumerate(temperatures):
    scaled_probas = softmax_with_temperature(next_token_logits, T)
    ax.bar(x + i * bar_width, scaled_probas, bar_width, label=f'Temperature={T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# 测试Top-K采样生成
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M.context_length,
    top_k=25,
    temperature=1.4
)
print("带温度和Top-K的生成:", token_ids_to_text(token_ids, tokenizer))

# 保存模型
torch.save(model.state_dict(), "model.pth")
print("模型已保存到 model.pth")