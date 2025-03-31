from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
from flask_cors import CORS
import bcrypt
import os
import fitz  # PyMuPDF
from docx import Document
import tempfile
from qa_system import QASystem, KnowledgeManager
import json
from datetime import datetime
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'campus_qa'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['KNOWLEDGE_FILE'] = 'data/knowledge.json'
app.config['INDEX_PATH'] = 'data/faiss_index.pkl'

# 确保数据目录存在
os.makedirs('data', exist_ok=True)

# 初始化知识管理器和问答系统
knowledge_manager = KnowledgeManager(app.config)
qa_engine = QASystem(knowledge_manager)
mysql = MySQL(app)

@app.route('/')
def index():
    if 'student_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            # 获取表单数据
            student_id = request.form.get('student_id', '').strip()
            password = request.form.get('password', '').strip()

            # 参数校验
            if not student_id or not password:
                return jsonify({"status": "error", "message": "请输入学号和密码"}), 400

            # 数据库操作
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE student_id = %s", (student_id,))
            user = cur.fetchone()
            cur.close()

            # 用户存在性检查
            if not user:
                return jsonify({"status": "error", "message": "用户不存在"}), 401

            # 密码验证
            if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                session['student_id'] = user['student_id']
                session['logged_in'] = True
                return redirect(url_for('index'))
            else:
                return jsonify({"status": "error", "message": "密码错误"}), 401

        except Exception as e:
            app.logger.error(f"登录异常: {str(e)}")
            return jsonify({"status": "error", "message": "服务器异常"}), 500

    # GET请求显示登录页面
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/api/login', methods=['POST'])
def login_api():
    try:
        data = request.get_json()
        student_id = data.get('username', '').strip()
        password = data.get('password', '').strip()

        # 参数校验
        if not student_id or not password:
            return jsonify({"status": "error", "message": "请输入学号和密码"}), 400

        # 数据库操作
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE student_id = %s", (student_id,))
        user = cur.fetchone()
        cur.close()

        # 用户存在性检查
        if not user:
            return jsonify({"status": "error", "message": "用户不存在"}), 401

        # 密码验证
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['student_id'] = user['student_id']
            session['logged_in'] = True
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "密码错误"}), 401

    except Exception as e:
        app.logger.error(f"登录异常: {str(e)}")
        return jsonify({"status": "error", "message": "服务器异常"}), 500


@app.route('/api/logout', methods=['POST'])
def logout_api():
    session.clear()
    return jsonify({"status": "success"})


@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if 'document' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        # 保存临时文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 处理文档并获取文本块和向量
        chunks, vectors = knowledge_manager.process_document(filepath, file.filename)
        
        # 更新知识库
        knowledge_manager.update_knowledge(chunks, vectors)

        # 保存到数据库
        cur = mysql.connection.cursor()
        for chunk in chunks:
            cur.execute("""
                INSERT INTO document_chunks 
                (filename, content, chunk_type, chunk_number, student_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                file.filename,
                chunk['content'],
                chunk['type'],
                chunk.get('page_number', chunk.get('paragraph_number', chunk.get('line_number'))),  # 这里补上了缺失的括号
                session['student_id'],
                json.dumps(chunk, ensure_ascii=False)
            ))
        mysql.connection.commit()

        # 生成预览内容
        preview = ""
        for chunk in chunks[:3]:  # 只显示前3个chunk
            preview += chunk['content'] + "\n"

        return jsonify({
            "status": "success",
            "preview": preview[:500] + "..." if len(preview) > 500 else preview,
            "chunks_count": len(chunks)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/knowledge', methods=['GET'])
def get_knowledge():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM knowledge_base ORDER BY created_at DESC")
        results = cur.fetchall()
        return jsonify({"status": "success", "knowledge": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/knowledge/<id>', methods=['DELETE'])
def delete_knowledge(id):
    if 'student_id' not in session or session.get('role') != 'admin':
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM knowledge_base WHERE id = %s", (id,))
        mysql.connection.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({"status": "error", "message": "Empty question"}), 400

        # 搜索相似文本块
        similar_chunks = knowledge_manager.search_similar_chunks(question)
        
        # 生成回答
        answer = qa_engine.generate_answer(question)

        # 保存搜索历史
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO search_history (student_id, query, answer, sources)
            VALUES (%s, %s, %s, %s)
        """, (session['student_id'], question, answer, 
              json.dumps([chunk['source'] for chunk in similar_chunks[:3]], ensure_ascii=False)))
        mysql.connection.commit()

        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": [chunk['source'] for chunk in similar_chunks[:3]]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT query as question, answer, sources, timestamp 
            FROM search_history 
            WHERE student_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 20
        """, (session['student_id'],))
        results = cur.fetchall()
        
        # 处理JSON字符串
        for result in results:
            if result['sources']:
                result['sources'] = json.loads(result['sources'])
            else:
                result['sources'] = []
                
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/check-auth')
def check_auth():
    if 'student_id' in session and session.get('logged_in'):
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Unauthorized"}), 401


if __name__ == '__main__':
    app.run(debug=True)
