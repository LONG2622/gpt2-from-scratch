#通过微调遵循人类指令6.5
#先为有监督微调准备数据集
import json
import os
import urllib
from importlib.metadata import version

pkgs = [
    "numpy",
    "matplotlib",
    "tiktoken",
    "torch",
    "tqdm",
    "tensorflow",
]
for p in pkgs:
    print(f"{p} version:{version(p)}")
    
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.utlopen(url) as response:
            text_data= response.read().decode("utf-8")
        with open(file_path, "w", encoding = "stf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
file_path = "instruction-data.json"
url = (
    "https://raw.githubuesercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json" 
)
data = download_and_load_file(file_path, url)
print(len(data))
#实现提示词格式函数
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task."
        f"Write a response that approciately completes the request"
        f"\n\n###Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else " "
    )
    return instruction_text + input_text
model_input = format_input(data[50])
desired_response = f"\n\n### REsponse:\n{data[20]['output']}"
#划分数据集
train_portion = int(len(data)*0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = [train_portion + test_portion]
#数据集分批
#实现指令数据集类
import torch
from torch.utils.data import Dataset
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_tests[index]
    def __len__(self):
        return len(self.data)
#自定义的聚合函数
def custom_collate_draft_1(
        batch, pad_token_id = 50356,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded=(
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch=(
    inputs_1,
    inputs_2,
    inputs_3
)
print(custom_collate_draft_1(batch))
def custom_collate_draft_2(
        batch, pad_token_id = 50356,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst , targets_lst= [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded=(
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index = -100,
        allowed_max_length=None,
        device="cpu"):
 #Find the longest sequence in the batch
# and increase the max length by +1, which will add one extra# padding token below
    batch_max_length = max(len(item)+1 for item in batch)
# Pad and prepare inputs
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
# Add an <|endoftext|> token
        new_item += [pad_token_id]
# Pad sequences to batch_max_length
        padded = (
        new_item + [pad_token_id] * 
        (batch_max_length - len(new_item))
        )
# Via padded[:-1], we remove the extra padded token
# that has been added via the +1 setting in batch_max_length
#(the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
# New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
#创建指令数据集的数据加载器
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor , targets_tensor 
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
targets_1  = torch.tensor([0,1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

from functools import partial
customized_collate_fn = partial(
    custom_collate_fn, 
    device = device,
    allowed_max_length = 1024
)
#初始化Dataloader
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
    )
val_dataset = InstructionDataset(val_data,tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
    )
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
#加载预训练的大语言模型
from gpt_download import download_and_load_gpt2
from ch03 import GPTModel
from ch04 import load_weights_into_gpt
BASE_CONFIG={
    "vocab_size":50257,
    "context_length":1024,
    "drop_rate":0.0,
    "qkv_bias":True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size = model_size,
    models_dir = "gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

#使用generate生成回复
from ch04 import generate, text_to_token_ids, token_ids_to_text
token_ids = generate(
    model =model,
    idx = text_to_token_ids(input_text, tokenizer),
    max_new_tokens = 35,
    context_size = BASE_CONFIG["context_length"]
    eos_id = 50256,
) 
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
print(response_text)
from ch04 import (calc_loss_loader,
                  train_model_simple)
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss= calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )
#对预训练的大模型进行指令微调
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr = 5e-5, weight_decay = 0.1)
num_epochs = 2
train_losses,val_losses, tokens_seen=train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq = 5,
    eval_iter = 5, start_context= format_input(val_data[0]), tokenizer = tokenzier
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) /60
#查看训练集损失曲线和验证曲线
from ch04 import plot_losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
