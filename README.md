build-a-LLM-from-scratch
从零手写复现 GPT-2 大模型项目，参考《从零构建大语言模型》全书源码，完整实现词嵌入、因果自注意力、Transformer 堆叠、预训练、Top-k + 温度采样生成、下游文本分类微调全链路。
一、环境配置
1. Python 版本
参考文档推荐：Python 3.9.6
项目实测环境：Python3.13.9，高版本可完美兼容 PyTorch 与项目依赖，无版本报错，无需降级 Python。
2. 基于 uv 快速搭建虚拟环境（推荐轻量化包管理器）
powershell
# 1. 全局安装uv包管理工具
pip install uv

# 2. 在项目根目录创建独立虚拟环境
uv venv .venv --python=3.13.9

# 3. Windows Powershell激活虚拟环境
.venv\Scripts\Activate.ps1

# 4. 安装PyTorch核心框架
uv pip install torch

# 5. 一键批量安装项目全部依赖（目录需存在requirements.txt）
uv pip install -r .\requirements.txt
二、项目内容说明
ch01~ch06 系列源码：分章节实现
分词预处理、GPT 词表编码
单层因果自注意力、多头注意力
Transformer 编码器模块、完整 GPT-2 124M 架构搭建
无标签小说文本预训练、损失收敛、自定义文本生成（温度缩放 + Top-K 采样）
基于预训练 GPT 微调垃圾短信二分类任务
the-verdict.txt：预训练用原始英文小说数据集
gpt_download.py：可选，用于拉取 OpenAI 官方 GPT-2 预训练权重（本地可选下载，不提交至代码仓库）
三、分词在线工具
Tiktoken 在线可视化分词工具：https://tiktokenizer.vercel.app/
用于调试 gpt2 词编码结果，直观查看文本→token-id 映射关系
四、仓库说明
.gitignore已配置：自动忽略预训练权重*.pth、GPT 官方权重文件夹gpt2/、数据集压缩包、缓存文件，仅上传源码
预训练权重、拆分后的 csv 数据集、解压后的 sms 数据集均保存在本地，不随代码提交 Github
五、运行指引
激活虚拟环境后，直接运行对应章节代码：python ch05.py
程序自动加载本地the-verdict.txt数据集，开始预训练；训练结束自动生成英文文本、执行分类推理。