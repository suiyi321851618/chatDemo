import os

os.environ["HF_REMOTECODE_DISABLE_SIGALRM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
import bcrypt
import tempfile
import faiss
import numpy as np
import torch
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.urandom(24),
    MYSQL_HOST='localhost',
    MYSQL_USER='root',
    MYSQL_PASSWORD='1234',
    MYSQL_DB='campus_qa',
    UPLOAD_FOLDER=tempfile.mkdtemp(),
    ALLOWED_EXTENSIONS={'pdf', 'docx', 'doc', 'txt'},
    KNOWLEDGE_FILE='data/knowledge.txt'
)

# 初始化MySQL
mysql = MySQL(app)

# 初始化AI模型
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/chatglm3-6b",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float32
).eval()


class KnowledgeManager:
    def __init__(self):
        self.knowledge = []
        self.index = None
        self.load_knowledge()

    def embed_text(self, texts):
        """生成文本嵌入向量"""
        inputs = tokenizer(texts, padding=True, truncation=True,
                           return_tensors="pt", max_length=256)
        # 设备对齐
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # 获取嵌入（使用logits均值作为文本表示）
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.logits.mean(dim=1).cpu().numpy()

        torch.cuda.empty_cache()  # 主动释放显存
        return embeddings

    def load_knowledge(self):
        """从文件加载知识库"""
        try:
            with open(app.config['KNOWLEDGE_FILE'], 'r', encoding='utf-8') as f:
                self.knowledge = [line.strip() for line in f if line.strip()]

            if self.knowledge:
                embeddings = self.embed_text(self.knowledge)
                dimension = embeddings.shape
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings.astype('float32'))
        except Exception as e:
            app.logger.error(f"加载知识库失败: {str(e)}")

    def update_knowledge(self, new_texts):
        """动态更新知识库"""
        try:
            # 更新文件
            with open(app.config['KNOWLEDGE_FILE'], 'a', encoding='utf-8') as f:
                f.write('\n'.join(new_texts) + '\n')

            # 更新内存数据
            new_embeddings = self.embed_text(new_texts)
            if self.index is None:
                dimension = new_embeddings.shape
                self.index = faiss.IndexFlatL2(dimension)
            self.index.add(new_embeddings.astype('float32'))
            self.knowledge.extend(new_texts)
        except Exception as e:
            app.logger.error(f"更新知识库失败: {str(e)}")


class QASystem:
    def __init__(self,km = KnowledgeManager()):
        self.km = km
        self.tokenizer = tokenizer
        self.model = model

    def generate_answer(self, question):
        """生成回答的核心逻辑"""
        try:
            # 知识检索
            query_embed = self.km.embed_text([question])
            distances, indices = self.km.index.search(query_embed.astype('float32'), 3)
            contexts = [self.km.knowledge[i] for i in indices]

            # 生成回答
            prompt = f"基于校园知识回答问题：\n" + '\n'.join(contexts) + f"\n\n问题：{question}"
            response, _ = model.chat(tokenizer, prompt, temperature=0.3, history=[])
            return response
        except Exception as e:
            app.logger.error(f"生成回答失败: {str(e)}")
            return "暂时无法回答这个问题"


# 初始化系统组件




if __name__ == '__main__':
    knowledge_manager = KnowledgeManager()
    qa_system = QASystem(knowledge_manager)
    # 交互式循环
    print("校园问答系统已启动（输入exit退出）")
    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() in {'exit', 'quit', 'stop'}:
            break
        if not user_input:
            continue

        print(f"\nChatGLM：{qa_system.generate_answer(user_input)}")
