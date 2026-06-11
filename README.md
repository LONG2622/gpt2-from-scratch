从零构建大语言模型（GPT-2）完整复现项目
基于 PyTorch 从零手写实现 GPT-2 完整架构
参考《从零构建大语言模型》官方源码，纯手工复现 Transformer、因果自注意力、预训练过程、文本生成、下游分类微调，不直接调用现成大模型库，深入理解 LLM 底层核心原理。
 纯手写实现 &nbsp;|&nbsp;  完整训练流程 &nbsp;|&nbsp;  文本生成 &nbsp;|&nbsp;  下游任务微调
---
📌 项目简介
本项目完整复现小型 GPT-2 架构，从数据预处理、Tokenizer 编码、词嵌入、位置编码、多头自注意力、Transformer Block 堆叠，到模型预训练、Loss 收敛、Top-K / Temperature 采样文本生成、短信分类微调，全程从零实现。
适合深度学习、大模型入门学习，彻底吃透 GPT 生成式大模型底层逻辑。
实现功能
- 文本预处理、GPT2-Tokenizer 分词编码
- 因果掩码自注意力、多头注意力机制
- 完整 Transformer 解码器架构搭建
- 语言模型预训练（文本续写、上下文学习）
- Temperature / Top-K 随机采样生成文本
- 预训练权重微调：垃圾短信二分类任务
- 训练损失可视化、模型保存与推理

---
💻 环境配置
1. 运行环境
- 书中标准环境：Python 3.9.6
- 本项目实测环境：Python 3.13.9（高版本完美兼容）
- 框架：PyTorch 最新稳定版
2. 使用 uv 搭建虚拟环境（极速包管理器）
推荐使用 uv 管理依赖，速度远超 pip。
# 全局安装 uv
pip install uv
# 创建虚拟环境
uv venv .venv --python=3.13.9

# 激活环境（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 安装 PyTorch
uv pip install torch

# 批量安装所有项目依赖
uv pip install -r requirements.txt

---
📁 项目文件结构
- ch01 ~ ch06.py：分章节从零实现大模型核心代码
- the-verdict.txt：预训练英文文本数据集
- gpt_download.py：可选下载官方 GPT2 预训练权重
- requirements.txt：项目依赖清单
- .gitignore：自动忽略权重、数据集、缓存大文件
说明：模型权重、数据集、缓存文件均本地保留，不上传远程仓库，保证代码库干净轻量化。
---
🚀 运行方式
激活虚拟环境后，直接运行对应章节源码：
python ch05.py

程序自动完成：数据加载 → 训练迭代 → 损失下降 → 自动文本生成 → 分类推理。

---
🔍 辅助工具
在线 GPT 分词可视化工具（辅助理解 Token 映射）：
https://tiktokenizer.vercel.app/
---
📈 项目亮点
- 全程手写：无封装大模型库，纯原生 PyTorch 实现
- 原理通透：彻底理解注意力机制、GPT 解码器、生成原理
- 训练完整：实现预训练 + 微调双流程
- 适配新版 Python：3.13 高版本完美运行无报错
- 仓库干净规范：忽略大文件，适合开源展示 & 简历项目
---
📄 开源说明
本项目仅用于学习研究，欢迎 Star、Fork、学习交流，禁止用于商业用途。
