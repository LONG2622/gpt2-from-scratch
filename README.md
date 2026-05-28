# build-a-LLM-from-scratch
基于《从零构建大模型》复现GPT的项目
1.构建Python环境
书中环境为python 3.9.6 本人使用的是3.13.9
作者建议落后几个版本（为适配pytorch包），但实际并不影响
2.设置uv（包管理器）和虚拟环境
pip install uv
虚拟环境（更安全）
 uv venv .venv --python=3.13.9
 激活：.venv\Scripts\Activate.ps1
 下载pytorch库：
uv pip install pytorch
下载该项目怒所需要的所有库：
uv pip install -r .\requirements.txt（注意：一定要在有该文本的目录下执行）

文本处理（数据预处理）
tiktokenizer(需魔法)
tiktokenizer.vercal.app
