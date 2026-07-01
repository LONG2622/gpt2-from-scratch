#ch01 -> data_preprocessing.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
# ===================== 1. 下载数据集 =====================
if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt" , "r", encoding = "UTF-8") as f:
    raw_text = f.read()
#raw_text 指令会显示verdict原文
#检验字数
len(raw_text)
#随便检验一下，预览print(raw_text[:99])
text = "hello , worlds. This is a test."
result = re.split(r'(\s)' , text)
print(result)
#复杂符号的处理
text = "hello , world . Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#删除空白冗余字符
result = [item.strip() for item in result if item.strip()]
print(result)
preprocessed = result
len(preprocessed)

#将这些token（词元）转化为 tokenID
#建立词汇表(构建vocabulary)
#先排序
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: i for i, token in enumerate(all_tokens)}

#构建实际词汇表(类似于字典排序)
vocab = {token :integer for integer,token in enumerate(all_words)}
for i,item in  enumerate(vocab.items()):
    print(item)
    if i >=50:
        break
#实现简单分词器（已优化可处理未知单词）
class SimpletokenizerV2:
    def __init__(self , vocab):
        self.str_to_int = vocab
        self.int_to_str = {i :s for s , i in vocab.items() }
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replace spaces before the specified punctuations(在指定的标点符号前替换空格)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
#编码示例：
custom_tokenizer = SimpletokenizerV2(vocab)

# ===================== 4. GPT2 官方 BPE 分词器 =====================
print("tiktoken 版本:", version("tiktoken"))
gpt2_tokenizer = tiktoken.get_encoding("gpt2")  # 固定用这个做后续训练

# 测试带特殊token的编码
test_text = "Hello , do you like tea?<|endoftext|> In the sunlit terraces."
integers = gpt2_tokenizer.encode(test_text, allowed_special={"<|endoftext|>"})
print("GPT2 编码结果:", integers)
#滑动窗口法数据采样
with open("the-verdict.txt","r",encoding = "utf-8") as f:
    raw_text = f.read()
enc_text = gpt2_tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x:{x}")
print(f"y:   {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(gpt2_tokenizer.decode(context), "---->", gpt2_tokenizer.decode([desired]))
#结果：
# and ----> established    
# and established ----> himself
# and established himself ----> in
# and established himself in ----> a

#数据加载器（dataloader)
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
#用于批量生成收入目标对的dataloader
def create_dataloader_v1(txt , 
                         tokenizer= None,
                         batch_size= 4 ,
                         max_length = 256,stride = 128,
                           shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last ,
        num_workers = num_workers
    )
    return dataloader
dataloader = create_dataloader_v1(raw_text , batch_size = 1,
                                  max_length = 4,stride = 1,shuffle =False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#步幅为4 的数据加载器采样
dataloader = create_dataloader_v1(
    raw_text,
    gpt2_tokenizer,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Input:\n", inputs)
print("outputs:\n", targets)

#创建token嵌入
inputs_ids = torch.tensor([2, 3, 5, 1])
#创建嵌入层
vocab_size  = 6
output_dim= 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size , output_dim)
print(embedding_layer.weight)

embedding_layer(torch.tensor([3]))

#encoding word positions(编码单词位置信息)
vocab_size=50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length= 4

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
token_embeddings.shape

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length , output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)