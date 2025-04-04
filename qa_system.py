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
from torch.cuda.amp import GradScaler, autocast
from transformers import BitsAndBytesConfig

os.environ["HF_REMOTECODE_DISABLE_SIGALRM"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class KnowledgeManager:
    def __init__(self, config):
        self.config = config
        self.knowledge = []
        self.index = None
        self.chunk_mapping = []
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # 使用量化版本的ChatGLM3 - 6B模型
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            "THUDM/chatglm3-6b",
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,  # 使用 float16 减少显存占用
            output_hidden_states=True,
            max_memory={0: "5.0GB","cpu": "32.0GB"}
        ).eval()
        self.device = next(self.model.parameters()).device  # 获取模型所在设备
        self.load_knowledge()

    def embed_text(self, texts):
        """使用embeddings生成文本向量"""
        try:
            if not texts or not isinstance(texts, list):
                print("警告：输入文本为空或格式不正确")
                return None

            # 批量处理文本
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512).to(self.device)  # 将输入数据移到模型所在设备

            scaler = GradScaler()
            # 使用混合精度计算
            with autocast(dtype=torch.float16):
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

            # 释放中间计算结果占用的显存
            torch.cuda.empty_cache()

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
                        # 批量处理向量化，减小批处理大小为4
                        if len(vectors) % 8 == 0:
                            batch_texts = [chunk['content'] for chunk in chunks[-8:]]
                            batch_vectors = self.embed_text(batch_texts)
                            if batch_vectors is not None:
                                vectors.extend(batch_vectors)
                            # 释放这一批次向量化处理占用的显存
                            torch.cuda.empty_cache()
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
                        # 批量处理向量化，减小批处理大小为4
                        if len(vectors) % 4 == 0:
                            batch_texts = [chunk['content'] for chunk in chunks[-4:]]
                            batch_vectors = self.embed_text(batch_texts)
                            if batch_vectors is not None:
                                vectors.extend(batch_vectors)
                            # 释放这一批次向量化处理占用的显存
                            torch.cuda.empty_cache()
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
                            # 批量处理向量化，减小批处理大小为4
                            if len(vectors) % 4 == 0:
                                batch_texts = [chunk['content'] for chunk in chunks[-4:]]
                                batch_vectors = self.embed_text(batch_texts)
                                if batch_vectors is not None:
                                    vectors.extend(batch_vectors)
                                # 释放这一批次向量化处理占用的显存
                                torch.cuda.empty_cache()

            # 处理剩余的文本块
            if len(vectors) < len(chunks):
                remaining_texts = [chunk['content'] for chunk in chunks[len(vectors):]]
                remaining_vectors = self.embed_text(remaining_texts)
                if remaining_vectors is not None:
                    vectors.extend(remaining_vectors)
                # 释放处理剩余文本块向量化占用的显存
                torch.cuda.empty_cache()

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
                self.index = faiss.IndexFlatL2(4096)  # ChatGLM3的隐藏层维度为4096

            # 加载知识库文件
            if os.path.exists(self.config['KNOWLEDGE_FILE']):
                with open(self.config['KNOWLEDGE_FILE'], 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                    self.knowledge.extend(knowledge_data)

                    # 更新向量索引
                    if knowledge_data:
                        embeddings = self.embed_text(knowledge_data)
                        if embeddings is not None:
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

    def search_similar_chunks(self, query, top_k=3):
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
    def __init__(self, km=None, max_history_length=5):
        self.km = km
        self.tokenizer = km.tokenizer
        self.model = km.model
        self.device = km.device
        self.conversation_history = []  # 维护对话历史
        self.max_history_length = max_history_length  # 最多保留5轮对话

    def generate_answer(self, question):
        try:
            # 知识检索
            similar_chunks = self.km.search_similar_chunks(question, top_k=1)  # 增加检索结果数量

            # 构建prompt
            context_prompt = "基于以下知识回答问题：\n" + '\n'.join(
                [chunk['content'] for chunk in similar_chunks]) if similar_chunks else ""
            history_prompt = "\n".join(
                [f"用户：{u}\n助手：{a}" for u, a in self.conversation_history]) if self.conversation_history else ""

            # 合并prompt
            prompt = []
            if context_prompt:
                prompt.append(context_prompt)
            if history_prompt:
                prompt.append(f"\n对话历史：\n{history_prompt}")
            prompt.append(f"\n问题：{question}")
            prompt = '\n'.join(prompt)

            # 生成回答
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # ChatGLM3-6B最大输入长度
            ).to(self.device)

            with autocast(dtype=torch.float16):
                with torch.no_grad():
                    response = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=100  # 增加生成长度
                    )

            response = self.tokenizer.decode(response[0], skip_special_tokens=True)

            # 更新对话历史（仅保留用户问题和助手回答）
            self.conversation_history.append((question, response))
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history.pop(0)

            # 释放显存
            torch.cuda.empty_cache()

            return response
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return "暂时无法回答这个问题"

    # 新增重置对话历史的方法
    def reset_conversation_history(self):
        self.conversation_history = []


if __name__ == '__main__':
    # 配置
    config = {
        'KNOWLEDGE_FILE': 'data/knowledge.json',
        'INDEX_PATH': 'data/faiss_index.pkl'
    }

    # 初始化系统组件
    knowledge_manager = KnowledgeManager(config)
    qa_system = QASystem(knowledge_manager, max_history_length=5)  # 设置最大对话轮数

    # 交互式循环
    print("校园问答系统已启动（输入exit退出，输入reset重置对话）")
    while True:
        user_input = input("\n用户：").strip()
        if user_input.lower() == 'reset':
            qa_system.conversation_history = []
            print("对话历史已重置")
            continue
        if user_input.lower() in {'exit', 'quit', 'stop'}:
            break
        if not user_input:
            continue

        print(f"\nChatGLM：{qa_system.generate_answer(user_input)}")
