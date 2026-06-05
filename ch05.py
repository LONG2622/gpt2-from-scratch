#开始微调6.3
#指令微调和分类微调
#准备数据集：下载和解压
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import urllib.request
import zipfile
import os
import pandas as pd
from pathlib import Path
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_tile_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping downloadand extraction.")
        return 
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File Downloaded and saved as {data_file_path}")
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(
    data_file_path, sep = "\t", header = None, names = ["Label", "Text"]
)
df
print(df["Label"].value_counts())
#创建一个平衡的数据集
def create_balanced_dataset(df):
    num_spam = df[df["Label"]=="spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state = 123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

#划分数据集
def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac = 1, random_state = 123
    ).reset_index(drop = True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
train_df, validation_df, test_df= random_split(
    balanced_df, 0.7, 0.1)
# 把标签转为 0/1
train_df["Label"] = train_df["Label"].map({"ham": 0, "spam": 1})
validation_df["Label"] = validation_df["Label"].map({"ham": 0, "spam": 1})
test_df["Label"] = test_df["Label"].map({"ham": 0, "spam": 1})

# 保存成 CSV 文件，给后面的 DataLoader 使用
train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)
# ==========================================================
#创建dateloader
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|encoder|>", allowed_special = {"<|endoftext|>"}))
#构建应该PytorchDataset类
import torch
from torch.utils.data import Dataset
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length= None,
                 pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts= [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts= [
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return(
            torch.tensor(encoded, dtype= torch.long),
            torch.tensor(label, dtype= torch.long)
        )
    def __len__ (self):
        return len(self.data)
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
train_dataset = SpamDataset(
    csv_file = "train.csv",
    max_length = None,
    tokenizer = tokenizer
)
print(train_dataset.max_length)
#将训练集和测试集填充到与最长训练序列匹配的长度
val_dataset = SpamDataset(
    csv_file ="validation.csv",
    max_length= train_dataset.max_length,
    tokenizer = tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length= train_dataset.max_length,
    tokenizer= tokenizer
)
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers= num_workers,
    drop_last = True, 
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers= num_workers,
    drop_last = False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    drop_last = False,
)

#6.4初始化带预训练权重模型
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
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
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
#加载预训练的GPT模型
from gpt_download import download_and_load_gpt2
from ch04 import GPTModel

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size = model_size, models_dir= "gpt2"
)
model = GPTModel(BASE_CONFIG)
model.eval()

from ch04 import text_to_token_ids, token_ids_to_text
from ch03 import generate_text_simple

text_1  = "Every effort moves you"
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_1, tokenizer),
    max_new_tokens= 15,
    context_size=BASE_CONFIG["context_length"]
    )
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'?Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $100 cash ot a $1000 reward.'"
)
token_ids =generate_text_simple(
    model = model, 
    idx = text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size= BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
#添加分类头
#先冻结模型
for param in model.parameters():
    param.requires_grad = False
#添加分类层
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features= BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
#解冻两个模块
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print(inputs)
print(inputs.shape)
#将tokenID传给模型
with torch.no_grad():
    outputs = model(inputs)
print(outputs)
#计算分类损失和准确率
#先实现微调中使用的模型评估函数
#准备数据集 ---> 模型设置 ---> 模型微调
#现在实现评估工具
def calc_accuracy_loader(data_loader, model, device, num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0,0

    if num_batches is None:
        num_batches= len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i ,(input_batch, target_batch) in enumerate (data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch= target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels= torch.argmax(logits, dim = -1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model ,device, num_batches = 10)
val_accuracy = calc_accuracy_loader(
    val_loader, model,device, num_batches=10
)
test_accuracy = calc_accuracy_loader(
    test_loader, model,device, num_batches=10
)
#开始lossloader
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
#计算分类损失
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate (data_loader):
            if i < num_batches:
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device
                )
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
#计算初始损失
with torch.no_grad():
    train_loss =calc_loss_loader(
        train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches = 5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches= 5)

#在有监督数据的基础上微调
#先微调处理垃圾消息分类
from ch02 import *
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
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs= [], [] , [], []
    examples_seen, global_step= 0, -1

    for epoch in  range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch , model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print({epoch+1}, {train_loss},{val_loss})
        train_accuracy = calc_accuracy_loader(
        train_loader, model ,device, num_batches = eval_iter)
        val_accuracy = calc_accuracy_loader(
        val_loader, model,device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 0.1)
num_epochs = 5
train_losses,val_losses, train_accs, val_accs, examples_seen= \
train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq = 50,
    eval_iter = 5
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) /60.
"""
#可视化
import matplotlib.pyplot as plt
def plot_values(
        epochs_seen, examples_seen, train_values, val_values,
        label = "loss" ):
    fig, ax1 = plt.subplots(figsize = (5,3))
    ax1.plot(epochs_seen, train_values, label = f"Training{label}" )
    ax1.plot(
        epochs_seen, val_values, linestyle = "-.",
        label = f"Validation{label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha = 0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
example_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen, train_losses, val_losses)
#绘制分类准确率图表
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor, example_seen_tensor, train_accs, val_accs,
    label="accuracy" 
)
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy= calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
#6.5垃圾消息分类器
def classify_review(
    text, model, tokenizer, device, max_length = None,
    pad_token_id = 50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(
        max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(
            input_ids, device =device
        ).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim = -1).item()
    return "spam" if predicted_label == 1 else "not spam"

text_1 = (
    "You are a winner you have been specially"
    " selected to recieve $100 cash or a $2000 reward."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length = train_dataset.max_length 
))
#保存模型
torch.save(model.state_dict(),"review_classifier.pth")
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)
"""
# 可视化
import matplotlib.pyplot as plt
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 画损失曲线
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 第二条 x 轴
    ax2 = ax1.twiny()
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.show()
# 画图
if len(train_losses) > 0 and len(val_losses) > 0:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_tensor, None, train_losses, val_losses)