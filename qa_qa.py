import os

os.environ["HF_REMOTECODE_DISABLE_SIGALRM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型量化配置（当前未启用）
quantization_config = {
    # 'load_in_4bit': True,
    # 'bnb_4bit_compute_dtype': torch.float16,
    # 'bnb_4bit_use_double_quant': True
}


class CampusQA:
    def __init__(self):
        """初始化问答系统"""
        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b",
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm3-6b",
            trust_remote_code=True,
            # quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float32
        )
        self.model.eval()  # 设置为评估模式

        # 加载知识库并构建索引
        self.knowledge, self.index = self.load_knowledge("data/knowledge.txt")

    def embed_text(self, texts):
        """生成文本嵌入向量"""
        # 文本编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        )

        # 设备对齐
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 获取嵌入（使用logits均值作为文本表示）
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.logits.mean(dim=1).cpu().numpy()

        torch.cuda.empty_cache()  # 主动释放显存
        return embeddings

    def load_knowledge(self, file_path):
        """加载知识库并构建向量数据库"""
        # 读取知识条目
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        # 生成嵌入并构建索引
        embeddings = self.embed_text(texts)
        dimension = embeddings.shape[1]

        # 创建L2距离索引
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        return texts, index

    def retrieve(self, query, top_k=3):
        """检索最相关的知识条目"""
        query_embed = self.embed_text([query])
        distances, indices = self.index.search(
            query_embed.astype('float32'),
            top_k
        )
        return [self.knowledge[i] for i in indices[0]]

    def generate_answer(self, question):
        """生成最终答案"""
        # 检索相关知识
        contexts = self.retrieve(question)

        # 构建提示模板
        prompt = (
            f"基于以下校园知识回答问题：\n"
            f"{chr(10).join(contexts)}\n\n"
            f"问题：{question}"
        )

        # 生成回答
        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                temperature=0.3,
                history=[]
            )
        return response


if __name__ == "__main__":
    # 初始化问答系统
    qa_system = CampusQA()

    # 交互式循环
    print("校园问答系统已启动（输入exit退出）")
    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in {'exit', 'quit', 'stop'}:
            break
        if not user_input:
            continue

        print(f"\nChatGLM：{qa_system.generate_answer(user_input)}")