"""
RAG系统 WebDemo
使用 Gradio 构建交互式问答界面
"""
# ==================== 环境变量修复（必须在所有第三方库导入之前） ====================
import os

# 设置国内镜像站（解决无法访问HuggingFace的问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

# 解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# SSL修复
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import tiktoken
import gradio as gr
from typing import Tuple, List

# 导入RAG模块
from RAG import create_rag_system
from config_clean import GPT_CONFIG_124M
from lora_adapter import LoRAGPTModel

# 全局变量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = tiktoken.get_encoding("gpt2")
RAG_SYSTEM = None
MODEL = None

def load_model():
    """
    加载预训练模型和RAG系统
    """
    global MODEL, RAG_SYSTEM
    print(f"使用设备: {DEVICE}")
    
    # 创建LoRA-GPT模型
    cfg = GPT_CONFIG_124M.copy()
    cfg["context_length"] = 512
    
    MODEL = LoRAGPTModel(
        cfg,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0
    )
    MODEL.to(DEVICE)
    
    # 加载预训练权重
    pretrained_path = "./gpt_sft_lora_model.pth"
    if os.path.exists(pretrained_path):
        print(f"加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=DEVICE)
        
        # 处理位置嵌入形状不匹配
        if "pos_emb.weight" in checkpoint:
            if checkpoint["pos_emb.weight"].shape[0] != cfg["context_length"]:
                checkpoint["pos_emb.weight"] = checkpoint["pos_emb.weight"][:cfg["context_length"], :]
        
        MODEL.load_state_dict(checkpoint)
        print("预训练模型加载成功!")
    else:
        print("未找到预训练模型，使用随机初始化权重")
    
    # 创建RAG系统
    RAG_SYSTEM = create_rag_system(
        model=MODEL,
        tokenizer=TOKENIZER,
        device=DEVICE,
        knowledge_dir="./knowledge",
        rebuild_index=False,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    print("RAG系统加载完成!")

def generate_answer(query: str, 
                    top_k: int = 3,
                    max_new_tokens: int = 128,
                    temperature: float = 0.7,
                    top_p: float = 0.9) -> Tuple[str, str]:
    """
    生成回答并返回检索到的文档
    """
    if not RAG_SYSTEM:
        return "RAG系统尚未加载，请等待...", ""
    
    try:
        print(f"\n[DEBUG] 查询: {query}")
        answer, retrieved_docs = RAG_SYSTEM.generate_answer(
            query=query,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        print(f"[DEBUG] 回答: {answer[:100]}...")
        
        # 格式化检索到的文档
        doc_text = ""
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            doc_text += f"📄 文档{i} (相似度: {score:.2f})\n{preview}\n\n"
        
        return answer, doc_text
    
    except Exception as e:
        import traceback
        full_error = f"生成出错: {str(e)}\n\n{traceback.format_exc()}"
        print(f"[ERROR] {full_error}")
        return f"生成出错: {str(e)}", ""

# 创建Gradio界面
with gr.Blocks(title="RAG问答系统", theme=gr.themes.Soft()) as demo:
    # 标题
    gr.Markdown("""
    # 🤖 RAG检索增强生成问答系统
    
    基于GPT-2 + LoRA + Sentence-Transformers构建的检索增强问答系统
    
    **使用方式**：在下方输入框中提出问题，系统会从知识库中检索相关文档并生成回答
    """)
    
    # 输入区域
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="📝 输入问题",
                placeholder="请输入您的问题，例如：什么是LoRA？",
                lines=3,
                max_lines=10
            )
        
        with gr.Column(scale=1):
            # 参数设置
            top_k = gr.Slider(
                label="检索文档数量",
                minimum=1,
                maximum=5,
                value=3,
                step=1
            )
            
            max_new_tokens = gr.Slider(
                label="最大生成长度",
                minimum=32,
                maximum=512,
                value=128,
                step=32
            )
            
            temperature = gr.Slider(
                label="温度系数 (多样性)",
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1
            )
            
            top_p = gr.Slider(
                label="Top-p采样",
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1
            )
    
    # 按钮
    submit_btn = gr.Button("🚀 生成回答", variant="primary", size="lg")
    
    # 输出区域
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="💡 回答",
                lines=8,
                max_lines=20,
                interactive=False
            )
        
        with gr.Column(scale=1):
            docs_output = gr.Textbox(
                label="📚 检索到的文档",
                lines=8,
                max_lines=20,
                interactive=False
            )
    
    # 示例问题
    gr.Markdown("""
    ### 💡 示例问题
    """)
    examples = gr.Examples(
        examples=[
            ["什么是LoRA？", 3, 128, 0.7, 0.9],
            ["Transformer的核心组件有哪些？", 3, 128, 0.7, 0.9],
            ["GPT是如何生成文本的？", 3, 128, 0.7, 0.9],
            ["什么是自注意力机制？", 3, 128, 0.7, 0.9]
        ],
        inputs=[query_input, top_k, max_new_tokens, temperature, top_p],
        outputs=[answer_output, docs_output],
        fn=generate_answer,
        cache_examples=False
    )
    
    # 绑定事件
    submit_btn.click(
        fn=generate_answer,
        inputs=[query_input, top_k, max_new_tokens, temperature, top_p],
        outputs=[answer_output, docs_output]
    )
    
    # 快捷键
    query_input.submit(
        fn=generate_answer,
        inputs=[query_input, top_k, max_new_tokens, temperature, top_p],
        outputs=[answer_output, docs_output]
    )

# 启动前加载模型
print("正在加载模型...")
load_model()
# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=True             # 是否生成共享链接
    )