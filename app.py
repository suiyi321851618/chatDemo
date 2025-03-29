from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
import bcrypt
import os
import fitz  # PyMuPDF
from docx import Document
import tempfile
# from qa_system import QASystem, KnowledgeManager
from qa_qa import CampusQA
# from datetime import datetime
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'campus_qa'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# qa_engine = QASystem(KnowledgeManager())
# qa_engine = QASystem()
qa_engine = CampusQA()
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


@app.route('/api/ask', methods=['POST'])
def ask_question():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({"status": "error", "message": "Empty question"}), 400

        # 优先检查文档内容
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT content FROM documents 
            WHERE student_id = %s 
            ORDER BY uploaded_at DESC 
            LIMIT 1
        """, (session['student_id'],))
        doc_result = cur.fetchone()

        # 检查知识库
        cur.execute("""
            SELECT content FROM knowledge_base 
            WHERE MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE)
            LIMIT 1
        """, (question,))
        kb_result = cur.fetchone()

        # 生成回答
        answer = None
        if doc_result and question.lower() in doc_result['content'].lower():
            answer = f"根据您上传的文档：{doc_result['content'][:200]}..."
        elif kb_result:
            answer = f"知识库答案：{kb_result['content']}"
        else:
            # 调用原始QA引擎
            answer = qa_engine.generate_answer(question)

        # 保存搜索历史
        cur.execute("""
            INSERT INTO search_history (student_id, query)
            VALUES (%s, %s)
        """, (session['student_id'], question))
        mysql.connection.commit()

        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        # 保存临时文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 解析文档内容
        content = ''
        if file.filename.endswith('.pdf'):
            doc = fitz.open(filepath)
            for page in doc:
                content += page.get_text()
        elif file.filename.endswith(('.doc', '.docx')):
            doc = Document(filepath)
            content = '\n'.join([para.text for para in doc.paragraphs])
        else:  # 文本文件
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

        # 保存到数据库
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO documents (filename, content, student_id)
            VALUES (%s, %s, %s)
        """, (file.filename, content, session['student_id']))
        mysql.connection.commit()

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "content_length": len(content)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/knowledge', methods=['GET', 'POST', 'DELETE'])
def manage_knowledge():
    if 'student_id' not in session or session.get('role') != 'admin':
        return jsonify({"status": "error", "message": "Forbidden"}), 403

    cur = mysql.connection.cursor()

    if request.method == 'GET':
        cur.execute("SELECT * FROM knowledge_base ORDER BY created_at DESC")
        results = cur.fetchall()
        return jsonify({"status": "success", "data": results})

    elif request.method == 'POST':
        data = request.get_json()
        try:
            cur.execute("""
                INSERT INTO knowledge_base (title, content, author)
                VALUES (%s, %s, %s)
            """, (data['title'], data['content'], session['student_id']))
            mysql.connection.commit()
            return jsonify({"status": "success", "id": cur.lastrowid})
        except KeyError:
            return jsonify({"status": "error", "message": "Missing fields"}), 400

    elif request.method == 'DELETE':
        entry_id = request.args.get('id')
        if not entry_id:
            return jsonify({"status": "error", "message": "Missing ID"}), 400
        cur.execute("DELETE FROM knowledge_base WHERE id = %s", (entry_id,))
        mysql.connection.commit()
        return jsonify({"status": "success"})


@app.route('/api/history')
def get_history():
    if 'student_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT query, timestamp 
        FROM search_history 
        WHERE student_id = %s 
        ORDER BY timestamp DESC 
        LIMIT 20
    """, (session['student_id'],))
    results = cur.fetchall()
    return jsonify({"status": "success", "data": results})


if __name__ == '__main__':
    app.run(debug=True)
