-- 创建文档块表
CREATE TABLE IF NOT EXISTS document_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    chunk_type ENUM('page', 'paragraph', 'line') NOT NULL,
    chunk_number INT NOT NULL,
    student_id VARCHAR(50) NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(student_id)
);

-- 修改搜索历史表
ALTER TABLE search_history
ADD COLUMN sources JSON AFTER answer;

-- 创建索引
CREATE INDEX idx_filename ON document_chunks(filename);
CREATE INDEX idx_student_id ON document_chunks(student_id);
CREATE INDEX idx_chunk_type ON document_chunks(chunk_type);
CREATE INDEX idx_created_at ON document_chunks(created_at);

-- 删除旧的documents表（如果存在）
DROP TABLE IF EXISTS documents; 