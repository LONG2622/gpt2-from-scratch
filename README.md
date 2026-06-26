# 从零构建大语言模型（GPT-2）完整项目

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 基于PyTorch从零手写实现GPT-2完整架构，涵盖LoRA微调、SFT指令微调、RAG检索增强生成等完整技术栈

## 📌 项目简介

本项目完整复现GPT-2架构，并扩展实现了当前大模型领域的核心技术：

- **基础架构**：从数据预处理到GPT-2模型完整实现
- **参数高效微调**：LoRA（Low-Rank Adaptation）实现
- **指令微调**：SFT（Supervised Fine-Tuning）实现
- **检索增强生成**：RAG（Retrieval-Augmented Generation）实现

适合深度学习、大模型入门学习，彻底吃透LLM底层核心原理。

## ✨ 功能特性

### 核心功能

| 模块 | 功能 | 说明 |
|------|------|------|
| **数据处理** | 文本分词、滑动窗口采样 | 支持GPT-2 BPE分词器 |
| **注意力机制** | 多头自注意力、因果掩码 | 实现Transformer核心组件 |
| **GPT模型** | 完整GPT-2架构 | 包含12层Transformer块 |
| **文本生成** | Temperature/Top-p采样 | 支持多样化文本生成 |
| **LoRA微调** | 参数高效微调 | 仅训练0.1%参数 |
| **SFT微调** | 指令遵循能力 | 支持Instruction格式数据 |
| **RAG系统** | 检索增强生成 | 集成FAISS向量数据库 |

### 技术亮点

- ✅ **纯手写实现**：无封装大模型库，纯原生PyTorch实现
- ✅ **原理通透**：彻底理解注意力机制、GPT解码器、生成原理
- ✅ **训练完整**：预训练 + LoRA + SFT + RAG完整流程
- ✅ **高版本兼容**：Python 3.13完美运行
- ✅ **工业级代码**：模块化设计、注释完善、易于扩展

## 💻 环境配置

### 运行环境

- **Python**: 3.13+
- **PyTorch**: 最新稳定版
- **CUDA**: 11.8+（可选，用于GPU加速）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/your-username/build-a-LLM-from-scratch.git
cd build-a-LLM-from-scratch
```

#### 2. 创建虚拟环境

**使用 uv（推荐）**：

```bash
# 安装 uv
pip install uv

# 创建虚拟环境
uv venv .venv --python=3.13

# 激活环境（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 激活环境（Linux/Mac）
source .venv/bin/activate
```

**使用 conda**：

```bash
conda create -n llm python=3.13
conda activate llm
```

#### 3. 安装依赖

```bash
# 安装PyTorch（根据你的CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio

# 安装项目依赖
pip install -r requirements.txt
```

#### 4. 下载预训练权重（可选）

```bash
# 方式1：从HuggingFace下载（需要网络）
python gpt_download.py

# 方式2：手动下载后放置到 ./gpt2_pretrained/
# 下载地址：https://huggingface.co/gpt2
```

## 🚀 使用指南

### 1. 运行GPT-2基础模型

```bash
python ch03.py
```

### 2. 运行LoRA微调

```bash
python lora.py
```

### 3. 运行SFT指令微调

```bash
python sft_finetune.py
```

### 4. 运行RAG系统

```bash
# 首次运行会自动下载嵌入模型
python RAG.py

# 设置HF镜像加速下载（可选）
# Windows PowerShell
set HF_ENDPOINT=https://hf-mirror.com
python RAG.py

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
python RAG.py
```

## 📁 项目结构

```
build-a-LLM-from-scratch/
├── ch01.py              # 数据处理模块
├── ch02.py              # 注意力机制实现
├── ch03.py              # GPT-2模型架构
├── ch04.py              # 文本生成
├── ch05.py              # 微调任务
├── ch06.py              # 其他功能
├── lora.py              # LoRA实现
├── sft_finetune.py      # SFT实现
├── RAG.py               # RAG实现
├── config_clean.py      # 配置文件
├── gpt_download.py      # 权重下载脚本
├── requirements.txt     # 依赖清单
├── .gitignore           # Git忽略配置
└── README.md            # 项目说明

# 不上传的文件
./gpt2_pretrained/       # GPT-2预训练权重
./knowledge/             # RAG知识库
./gpt_sft_lora_model.pth # 微调模型
./__pycache__/           # Python缓存
```

## 📚 学习路径

建议按以下顺序学习：

1. **ch01.py** - 理解数据处理流程
2. **ch02.py** - 掌握注意力机制
3. **ch03.py** - 构建GPT-2模型
4. **lora.py** - 学习参数高效微调
5. **sft_finetune.py** - 理解指令微调
6. **RAG.py** - 掌握检索增强生成

## 🔧 配置说明

### 模型配置

编辑 `config_clean.py` 修改模型配置：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### LoRA配置

在 `lora.py` 中调整LoRA参数：

```python
lora_rank = 8          # 低秩维度
lora_alpha = 8         # 缩放因子
lora_dropout = 0.0     # Dropout率
```

## 📊 性能指标

| 模型 | 参数量 | 训练时间 | 显存占用 |
|------|--------|----------|----------|
| GPT-2 124M | 124M | ~2h | ~2GB |
| LoRA微调 | 0.1% | ~10min | ~2.5GB |
| SFT微调 | 0.1% | ~5min | ~2.5GB |
| RAG推理 | - | <1s | ~3GB |

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- 参考书籍：《从零构建大语言模型》
- GPT-2: OpenAI
- Sentence-Transformers: HuggingFace

## 📮 联系方式

- 项目链接: [https://github.com/LONG2622/build-a-LLM-from-scratch](https://github.com/LONG2622/build-a-LLM-from-scratch)
- 问题反馈: [Issues](https://github.com/LONG2622/build-a-LLM-from-scratch/issues)

---

⭐ 如果这个项目对你有帮助，请给个 Star！
