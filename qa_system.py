import os
import json
import numpy as np
import torch
import faiss
import pickle
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel
from PyPDF2 import PdfReader
from docx import Document

os.environ["HF_REMOTECODE_DISABLE_SIGALRM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class KnowledgeManager:
    def __init__(self, config):
        self.config = config
        self.knowledge = []
        self.index = None
        self.chunk_mapping = []
        # 使用量化版本的ChatGLM3-6B模型
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            "THUDM/chatglm3-6b",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float32
        ).eval()
        self.load_knowledge()

    def embed_text(self, texts):
        """使用embeddings生成文本向量"""
        try:
            if not texts or not isinstance(texts, list):
                print("警告：输入文本为空或格式不正确")
                return None

            # 批量处理文本
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512)

            # 获取embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 检查输出
                if not hasattr(outputs, 'hidden_states') or not outputs.hidden_states:
                    print("警告：模型输出中没有隐藏状态")
                    return None

                # 使用最后一层的隐藏状态
                last_hidden_state = outputs.hidden_states[-1]
                if last_hidden_state is None or last_hidden_state.size(0) == 0:
                    print("警告：最后一层隐藏状态为空")
                    return None
                
                # 计算平均向量
                embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
                
                # 检查生成的向量
                if embeddings is None or embeddings.size == 0:
                    print("警告：生成的向量为空")
                    return None
                
                # 确保向量维度正确
                if embeddings.shape[1] != 4096:  # ChatGLM3的隐藏层维度
                    print(f"警告：向量维度不正确，期望4096，实际{embeddings.shape[1]}")
                    return None

            return embeddings
        except Exception as e:
            print(f"生成文本向量失败: {str(e)}")
            return None

    def process_document(self, filepath, filename):
        """处理文档并返回文本块和向量"""
        chunks = []
        vectors = []

        try:
            if filename.endswith('.pdf'):
                doc = PdfReader(filepath)
                for page_num, page in enumerate(doc.pages):
                    page_content = page.extract_text()
                    if page_content.strip():
                        chunks.append({
                            'type': 'page',
                            'page_number': page_num + 1,
                            'content': page_content,
                            'source': filename,
                            'created_at': datetime.now().isoformat()
                        })
                        # 批量处理向量化
                        if len(vectors) % 32 == 0:  # 每32个文本块处理一次
                            batch_texts = [chunk['content'] for chunk in chunks[-32:]]
                            batch_vectors = self.embed_text(batch_texts)
                            if batch_vectors is not None:
                                vectors.extend(batch_vectors)
            elif filename.endswith(('.doc', '.docx')):
                doc = Document(filepath)
                for para_num, para in enumerate(doc.paragraphs):
                    if para.text.strip():
                        chunks.append({
                            'type': 'paragraph',
                            'paragraph_number': para_num + 1,
                            'content': para.text,
                            'source': filename,
                            'created_at': datetime.now().isoformat()
                        })
                        # 批量处理向量化
                        if len(vectors) % 32 == 0:
                            batch_texts = [chunk['content'] for chunk in chunks[-32:]]
                            batch_vectors = self.embed_text(batch_texts)
                            if batch_vectors is not None:
                                vectors.extend(batch_vectors)
            else:  # 文本文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines):
                        if line.strip():
                            chunks.append({
                                'type': 'line',
                                'line_number': line_num + 1,
                                'content': line.strip(),
                                'source': filename,
                                'created_at': datetime.now().isoformat()
                            })
                            # 批量处理向量化
                            if len(vectors) % 32 == 0:
                                batch_texts = [chunk['content'] for chunk in chunks[-32:]]
                                batch_vectors = self.embed_text(batch_texts)
                                if batch_vectors is not None:
                                    vectors.extend(batch_vectors)

            # 处理剩余的文本块
            if len(vectors) < len(chunks):
                remaining_texts = [chunk['content'] for chunk in chunks[len(vectors):]]
                remaining_vectors = self.embed_text(remaining_texts)
                if remaining_vectors is not None:
                    vectors.extend(remaining_vectors)

        except Exception as e:
            raise Exception(f"文档处理错误: {str(e)}")

        return chunks, vectors

    def load_knowledge(self):
        """从文件加载知识库"""
        try:
            # 加载向量索引
            if os.path.exists(self.config['INDEX_PATH']):
                with open(self.config['INDEX_PATH'], 'rb') as f:
                    index_data = pickle.load(f)
                    self.index = index_data['index']
                    self.chunk_mapping = index_data['mapping']
                    self.knowledge = [chunk['content'] for chunk in self.chunk_mapping]
            else:
                # 初始化新的索引
                self.index = faiss.IndexFlatL2(768)  # 向量维度为768

            # 加载知识库文件
            if os.path.exists(self.config['KNOWLEDGE_FILE']):
                with open(self.config['KNOWLEDGE_FILE'], 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                    self.knowledge.extend(knowledge_data)

                    # 更新向量索引
                    if knowledge_data:
                        embeddings = self.embed_text(knowledge_data)
                        self.index.add(embeddings.astype('float32'))

        except Exception as e:
            print(f"加载知识库失败: {str(e)}")

    def save_index(self):
        """保存向量索引和映射关系"""
        try:
            with open(self.config['INDEX_PATH'], 'wb') as f:
                pickle.dump({
                    'index': self.index,
                    'mapping': self.chunk_mapping
                }, f)
        except Exception as e:
            print(f"保存索引失败: {str(e)}")

    def update_knowledge(self, chunks, vectors):
        """更新知识库"""
        try:
            # 更新向量索引
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                self.index.add(vectors_array)
                self.chunk_mapping.extend(chunks)
                self.save_index()

            # 更新知识库文件
            knowledge_data = [chunk['content'] for chunk in chunks]
            if os.path.exists(self.config['KNOWLEDGE_FILE']):
                with open(self.config['KNOWLEDGE_FILE'], 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_data.extend(knowledge_data)
                    with open(self.config['KNOWLEDGE_FILE'], 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
            else:
                with open(self.config['KNOWLEDGE_FILE'], 'w', encoding='utf-8') as f:
                    json.dump(knowledge_data, f, ensure_ascii=False, indent=2)

            self.knowledge.extend(knowledge_data)
        except Exception as e:
            print(f"更新知识库失败: {str(e)}")

    def search_similar_chunks(self, query, top_k=5):
        """搜索相似文本块"""
        try:
            # 获取查询文本的向量表示
            query_vector = self.embed_text([query])[0]

            # 使用FAISS进行相似度搜索
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'),
                top_k
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunk_mapping):
                    chunk = self.chunk_mapping[idx]
                    # 计算余弦相似度
                    similarity = float(1 / (1 + distances[0][i]))
                    results.append({
                        'content': chunk['content'],
                        'source': chunk['source'],
                        'similarity': similarity,
                        'type': chunk['type'],
                        'position': chunk.get('page_number', chunk.get('paragraph_number', chunk.get('line_number')))
                    })
            return results
        except Exception as e:
            print(f"搜索相似文本块失败: {str(e)}")
            return []


class QASystem:
    def __init__(self, km=None):
        self.km = km
        # 使用KnowledgeManager中的tokenizer和model
        self.tokenizer = km.tokenizer
        self.model = km.model

    def generate_answer(self, question, context=None):
        """生成回答的核心逻辑"""
        try:
            # 知识检索
            similar_chunks = self.km.search_similar_chunks(question)

            # 构建提示词
            if similar_chunks:
                contexts = [chunk['content'] for chunk in similar_chunks]
                prompt = f"基于以下知识回答问题：\n" + '\n'.join(contexts) + f"\n\n问题：{question}"
            else:
                prompt = f"问题：{question}"

            # 生成回答
            response, _ = self.model.chat(self.tokenizer, prompt, temperature=0.3, history=[])
            return response
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return "暂时无法回答这个问题"


if __name__ == '__main__':
    # 配置
    config = {
        'KNOWLEDGE_FILE': 'data/knowledge.json',
        'INDEX_PATH': 'data/faiss_index.pkl'
    }

    # 初始化系统组件
    knowledge_manager = KnowledgeManager(config)
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