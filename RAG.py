#RAG 系统
#用于解决信息检索问题，解决知识时效性问题， 提供知识更新功能
#减少幻觉，提供可溯源性，最终降低微调成本

# ==================== 环境变量修复（必须在所有第三方库导入之前） ====================
# 解决Windows下OpenMP运行时冲突问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import torch
import numpy as np
import tiktoken 
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

#解决Windows 系统的ssl 问题
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#文档分块器
class DocumentChunker:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("gpt2")
    def chunk_text(self, text: str) -> List[str]:
        """
        对文本进行分块
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk))

            if end >= len(tokens):
                break
            start = end - self.chunk_overlap
        return chunks
    def chunk_file(self, file_path: str) -> List[str]:
        """
        对文件进行分块
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.chunk_text(text)
    
#文本嵌入编码器
class TextEmbeddingEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        try:
            from sentence_transformers import SentenceTransformer
            # 设置缓存目录
            model_kwargs = {}
            if cache_dir:
                model_kwargs["cache_folder"] = cache_dir
                print(f"使用自定义缓存目录: {cache_dir}")
            
            # 优先尝试加载本地模型
            local_paths = [
                model_name,  # 直接传入的路径
                f"./models/{model_name}",  # 项目内models目录
                os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"),  # 默认缓存路径
            ]
            
            loaded_from = None
            for local_path in local_paths:
                if os.path.exists(local_path):
                    try:
                        self.model = SentenceTransformer(local_path, **model_kwargs)
                        loaded_from = local_path
                        print(f"从本地加载模型: {local_path}")
                        break
                    except Exception as e:
                        continue
            
            # 如果本地没有，从HuggingFace下载（会自动缓存）
            if loaded_from is None:
                print(f"从HuggingFace加载模型: {model_name}")
                self.model = SentenceTransformer(model_name, **model_kwargs)
                print(f"模型加载成功（已缓存到本地）")
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"嵌入维度: {self.embedding_dim}")
        except ImportError:
            raise ImportError("Please install the SentenceTransformer library: pip install sentence-transformers")
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        对文本进行嵌入编码
        """
        return self.model.encode(texts, convert_to_numpy=True)
    def encode_query(self, query: str) -> np.ndarray:
        """
        对查询进行嵌入编码
        """
        return self.encode([query])[0]

#向量存储与检索
class VectorStore:
    def __init__(self, encoder: TextEmbeddingEncoder, use_faiss: bool = True):
        self.encoder = encoder
        self.use_faiss = use_faiss
        self.embeddings = None
        self.documents = []

        if use_faiss:
            try:
                import faiss
                self.faiss = faiss  # 保存faiss模块引用
                self.faiss_index = None
                print("FAISS 索引已初始化")
            except ImportError:
                print("未安装FAISS，使用纯numpy检索")
                self.use_faiss = False
    def add_documents(self, documents: List[str]):
        """
        添加文档到向量存储
        """
        self.documents = documents
        #向量化文档片段
        self.embeddings = self.encoder.encode(documents)

        if self.use_faiss and self.faiss_index is None:
            #创建FAISS 索引
            self.faiss_index = self.faiss.IndexFlatL2(self.encoder.embedding_dim)
        if self.use_faiss:
            self.faiss_index.add(self.embeddings)
    def save(self, dir_path: str):
        """
        保存向量存储
        """
        os.makedirs(dir_path, exist_ok=True)
        #保存文档和嵌入
        with open(os.path.join(dir_path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)
        np.save(os.path.join(dir_path, "embeddings.npy"), self.embeddings)
        
        #保存索引
        if self.use_faiss and self.faiss_index is not None:
            self.faiss.write_index(self.faiss_index, os.path.join(dir_path, "faiss_index"))
        
    def load(self, dir_path: str):
        """
        从目录加载向量存储
        """
        #加载文档和嵌入
        with open(os.path.join(dir_path, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.embeddings = np.load(os.path.join(dir_path, "embeddings.npy"))
        #加载索引
        if self.use_faiss:
            index_path = os.path.join(dir_path, "faiss_index")
            if os.path.exists(index_path):
                self.faiss_index = self.faiss.read_index(index_path)
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        从向量存储中检索与查询最相似的文档片段
        """
        query_embedding = self.encoder.encode_query(query).reshape(1, -1)
        
        if self.use_faiss and self.faiss_index is not None:
            #使用FAISS 索引检索
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    similarity = 1.0 / (1.0 + distances[0][i])
                    results.append((self.documents[idx], similarity))
        else:
            #使用余弦相似度检索
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(self.documents[idx], similarities[idx]) for idx in top_indices]
        return results
    
#RAG管道
class RAGPipeline:
    def __init__(self, vector_store: VectorStore, model,
                 tokenizer, 
                 device: str = "cpu",
                 max_context_length: int = 1024):
        self.vector_store = vector_store
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_length = max_context_length

        self.model.eval()

    def build_prompt(self, query: str, retrieved_docs: List[Tuple[str, float]]):
        """
        构建RAG提示（与SFT训练时的格式保持一致）
        
        训练时的格式：
        ### Instruction: xxx
        ### Input: xxx
        ### Response: xxx
        """
        #构建上下文部分（作为Input传入）
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            context_parts.append(f"文档{i}: {doc}")
        context_text = "\n".join(context_parts)
        
        #使用与训练时相同的格式
        prompt = f"""### Instruction: 根据提供的文档内容回答问题
### Input: {context_text}

用户问题：{query}
### Response: """
        return prompt
    def generate_answer(self, 
                        query: str,
                        top_k : int = 3,
                        max_new_tokens: int = 256,
                        temperature: float = 0.7,
                        top_p : float = 0.9
                        )-> Tuple[str, List[Tuple[str, float]]]:
        """
        RAG生成过程
        """
        #检索文档片段
        retrieved_docs = self.vector_store.retrieve(query, top_k)
        if not retrieved_docs:
            return "没有检索到相关文档片段", []
        #构建提示词
        prompt = self.build_prompt(query, retrieved_docs)
        #编码提示词
        encoded = self.tokenizer.encode(prompt, allowed_special = {"<|endoftext|>"},)
        idx = torch.tensor(encoded, device = self.device).unsqueeze(0)
        #生成回答
        with torch.no_grad():
            for _ in range(max_new_tokens):
                #限制上下文长度
                idx_cond = idx[:, -self.max_context_length:]
                logits = self.model(idx_cond)
                logits = logits[:, -1, :]

                #温度采样
                logits = logits/temperature
                probs = torch.softmax(logits, dim=-1)
                #Top_p 采样
                sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, 
                                                                     sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                #采样下一个token
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
                #遇到结束符，停止生成
                if idx_next.item() == 50256:
                    break
        #解码并提取回答
        full_text = self.tokenizer.decode(idx.squeeze(0).tolist())
        answer = full_text[len(prompt):].strip()
        return answer, retrieved_docs
#使用示例
def create_rag_system(
        model,
        tokenizer,
        device,
        knowledge_dir:str = "./knowledge",
        rebuild_index: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2"
) -> RAGPipeline:
    """
    创建RAG系统
    """
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
    encoder = TextEmbeddingEncoder(model_name=embedding_model)
    vector_store = VectorStore(encoder, use_faiss=True)
    # 检查是否需要重新构建索引
    index_path = os.path.join(knowledge_dir, "vector_store")
    
    if not rebuild_index and os.path.exists(index_path):
        # 加载已有的向量存储
        vector_store.load(index_path)
    else:
        # 重新构建索引
        print("正在构建知识库索引...")
        # 收集所有文档
        all_documents = []
        
        # 读取knowledge目录下的所有文本文件
        if os.path.exists(knowledge_dir):
            for filename in os.listdir(knowledge_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(knowledge_dir, filename)
                    chunks = chunker.chunk_file(file_path)
                    all_documents.extend(chunks)
                    print(f"  已处理: {filename} → {len(chunks)} 个片段")
        
        # 如果没有知识库，使用默认文档
        if not all_documents:
            print(" 未找到知识库文件，使用示例文档")
            default_text = """LoRA(Low-Rank Adaptation)是一种参数高效的微调技术。
它通过在预训练模型的注意力层中注入低秩矩阵来实现微调，而不是更新所有参数。
LoRA的核心思想是冻结预训练模型的权重，只训练低秩矩阵A和B。
这样可以大幅减少可训练参数的数量，降低显存占用，加速训练过程。

Transformer是一种基于自注意力机制的深度学习模型架构。
它由编码器和解码器组成，每个部分都包含多个Transformer块。
每个Transformer块包含多头注意力层和前馈神经网络。
自注意力机制允许模型在处理序列时关注不同位置的信息。

GPT（Generative Pre-trained Transformer）是一种基于Transformer架构的大型语言模型。
它采用了仅解码器的结构，通过自回归方式生成文本。
GPT通过在大规模文本数据上进行预训练，学习语言的统计规律和语义知识。
预训练后的GPT可以通过微调适应各种下游任务。"""
            all_documents = chunker.chunk_text(default_text)
        
        # 添加到向量存储
        vector_store.add_documents(all_documents)
        
        # 保存索引
        vector_store.save(index_path)
    
    # 创建RAG管道
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context_length=512
    )
    
    return rag_pipeline

#主程序演示
if __name__ == "__main__":
    print("=" * 60)
    print("RAG（检索增强生成）系统演示")
    print("=" * 60)
    
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 加载配置和模型
    from config_clean import GPT_CONFIG_124M
    from lora_adapter import LoRAGPTModel
    
    cfg = GPT_CONFIG_124M.copy()
    cfg["context_length"] = 512
    
    # 创建LoRA-GPT模型
    model = LoRAGPTModel(
        cfg,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0
    )
    model.to(device)
    
    # 尝试加载预训练权重（如果存在）
    pretrained_path = "./gpt_sft_lora_model.pth"
    if os.path.exists(pretrained_path):
        print(f"加载预训练模型: {pretrained_path}")
        # 加载权重并处理形状不匹配问题
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # 处理pos_emb.weight形状不匹配（预训练模型可能用的是1024，当前是512）
        if "pos_emb.weight" in checkpoint:
            if checkpoint["pos_emb.weight"].shape[0] != cfg["context_length"]:
                print(f"  调整位置嵌入形状: {checkpoint['pos_emb.weight'].shape} → ({cfg['context_length']}, {cfg['emb_dim']})")
                checkpoint["pos_emb.weight"] = checkpoint["pos_emb.weight"][:cfg["context_length"], :]
        
        model.load_state_dict(checkpoint)
        print("  预训练模型加载成功!")
    else:
        print("未找到预训练模型，使用随机初始化权重")
    
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 3. 创建RAG系统
    rag_system = create_rag_system(
        model=model,
        tokenizer=tokenizer,
        device=device,
        knowledge_dir="./knowledge",
        rebuild_index=False
    )
    
    # 4. 测试RAG问答
    test_queries = [
        "什么是LoRA?",
        "Transformer的核心组件有哪些?",
        "GPT是如何生成文本的?",
        "什么是自注意力机制?"
    ]
    
    print("\n" + "=" * 60)
    print(" 测试问答")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n 用户问题: {query}")
        
        answer, retrieved_docs = rag_system.generate_answer(
            query=query,
            top_k=3,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f" 回答: {answer}")
        
        # 打印检索到的文档
        print("\n 检索到的相关文档:")
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"  [{i}] 相似度: {score:.2f}\n    {preview}")
    
    print("\n" + "=" * 60)
    print("RAG系统测试完成!")
    print("=" * 60)