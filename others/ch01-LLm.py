import os
import re
import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
# ===================== 1. 下载数据集 =====================
if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "the-verdict.txt")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("文本总长度:", len(raw_text))
print("开头预览:", raw_text[:99])

# ===================== 2. 基础文本预处理 =====================
text = "hello , world . Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result

# ===================== 3. 自定义简易分词器 =====================
all_words = sorted(set(preprocessed))
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# 独立命名，不覆盖GPT2分词器
custom_tokenizer = SimpleTokenizerV2(vocab)

# ===================== 4. GPT2 官方 BPE 分词器 =====================
print("tiktoken 版本:", version("tiktoken"))
gpt2_tokenizer = tiktoken.get_encoding("gpt2")  # 固定用这个做后续训练

# 测试带特殊token的编码
test_text = "Hello , do you like tea?<|endoftext|> In the sunlit terraces."
integers = gpt2_tokenizer.encode(test_text, allowed_special={"<|endoftext|>"})
print("GPT2 编码结果:", integers)

# ===================== 5. 滑动窗口构建样本 =====================
enc_text = gpt2_tokenizer.encode(raw_text)
print("数据集总token数:", len(enc_text))

enc_sample = enc_text[50:]
context_size = 4
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(gpt2_tokenizer.decode(context), "---->", gpt2_tokenizer.decode([desired]))

# ===================== 6. GPT 数据集类 =====================
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

# ===================== 7. 数据加载器 =====================
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

# 小批量测试
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
print("第一个批次:", next(data_iter))

# 批量8测试
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
inputs, targets = next(iter(dataloader))
print("Inputs:\n", inputs)
print("Targets:\n", targets)

# ===================== 8. Token 嵌入 + 位置嵌入 =====================
vocab_size = 50257
output_dim = 256
token_emb = torch.nn.Embedding(vocab_size, output_dim)
pos_emb = torch.nn.Embedding(4, output_dim)  # 上下文长度4

# 取一批数据测试
inputs, _ = next(iter(dataloader))
token_embeddings = token_emb(inputs)
pos_embeddings = pos_emb(torch.arange(4))
input_embeddings = token_embeddings + pos_embeddings

print("Token嵌入形状:", token_embeddings.shape)
print("位置嵌入形状:", pos_embeddings.shape)
print("最终输入形状:", input_embeddings.shape)