<template>
  <div class="container">
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">校园智能问答</a>
        <div class="collapse navbar-collapse">
          <ul class="navbar-nav me-auto">
            <li class="nav-item">
              <a class="nav-link" href="#" @click="activeSection = 'qa'">智能问答</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="#" @click="activeSection = 'knowledge'">知识库管理</a>
            </li>
          </ul>
          <button class="btn btn-light" @click="handleLogout">注销</button>
        </div>
      </div>
    </nav>

    <!-- 智能问答模块 -->
    <div v-if="activeSection === 'qa'" id="qaSection">
      <div class="question-box">
        <div class="input-group mb-3">
          <input type="text" class="form-control" v-model="question" placeholder="请输入您的问题...">
          <button class="btn btn-primary" @click="askQuestion">提问</button>
        </div>
        <div v-if="answer" class="mt-3" id="answer">
          {{ answer }}
        </div>
      </div>

      <!-- 文档上传模块 -->
      <div class="document-box mt-4">
        <h4>文档解析</h4>
        <input type="file" class="form-control mb-3" @change="handleFileUpload" accept=".pdf,.doc,.docx,.txt">
        <div v-if="uploadStatus" class="upload-status mb-2">{{ uploadStatus }}</div>
        <div v-if="docPreview" class="doc-preview">
          <pre>{{ docPreview }}</pre>
        </div>
      </div>

      <!-- 搜索历史模块 -->
      <div class="history-box mt-4">
        <h4>搜索历史</h4>
        <div class="list-group">
          <div v-for="(item, index) in searchHistory" :key="index" class="list-group-item">
            <div class="d-flex justify-content-between">
              <div>
                <strong>问题：</strong>{{ item.question }}
                <br>
                <strong>答案：</strong>{{ item.answer }}
              </div>
              <small class="text-muted">{{ item.timestamp }}</small>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 知识库管理界面 -->
    <div v-if="activeSection === 'knowledge'" class="knowledge-operations">
      <button class="btn btn-primary" @click="fetchKnowledge">刷新列表</button>
      <button class="btn btn-success" @click="showAddKnowledgeModal">新增条目</button>
      <div class="list-group mt-3">
        <div v-for="(item, index) in knowledgeList" :key="index" class="list-group-item">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <h5>{{ item.title }}</h5>
              <p class="mb-0">{{ item.content }}</p>
            </div>
            <div>
              <button class="btn btn-sm btn-danger" @click="deleteKnowledge(item.id)">删除</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import type { SearchHistoryItem, KnowledgeItem } from '../types'
import '../assets/styles/home.css'

const router = useRouter()
const activeSection = ref('qa')
const question = ref('')
const answer = ref('')
const uploadStatus = ref('')
const docPreview = ref('')
const searchHistory = ref<SearchHistoryItem[]>([])
const knowledgeList = ref<KnowledgeItem[]>([])

const askQuestion = async () => {
  if (!question.value) return
  
  answer.value = '思考中...'
  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question.value })
    })

    const data = await response.json()
    if (data.status === 'success') {
      answer.value = data.answer
      // 添加到搜索历史
      searchHistory.value.unshift({
        question: question.value,
        answer: data.answer,
        timestamp: new Date().toLocaleString()
      })
    } else {
      answer.value = `出错了：${data.message}`
    }
  } catch (error) {
    answer.value = '网络请求失败'
  }
}

const handleFileUpload = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return

  uploadStatus.value = '上传中...'
  const formData = new FormData()
  formData.append('document', file)

  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    })

    const data = await response.json()
    if (data.status === 'success') {
      uploadStatus.value = '上传成功'
      docPreview.value = data.preview
    } else {
      uploadStatus.value = `上传失败：${data.message}`
    }
  } catch (error) {
    uploadStatus.value = '上传失败，请稍后重试'
  }
}

const fetchKnowledge = async () => {
  try {
    const response = await fetch('/api/knowledge')
    const data = await response.json()
    if (data.status === 'success') {
      knowledgeList.value = data.knowledge
    }
  } catch (error) {
    console.error('获取知识库列表失败：', error)
  }
}

const fetchHistory = async () => {
  try {
    const response = await fetch('/api/history')
    const data = await response.json()
    if (data.status === 'success') {
      searchHistory.value = data.data.map((item: any) => ({
        question: item.question,
        answer: item.answer,
        timestamp: new Date(item.timestamp).toLocaleString()
      }))
    }
  } catch (error) {
    console.error('获取历史记录失败：', error)
  }
}

const showAddKnowledgeModal = () => {
  // TODO: 实现添加知识库条目的模态框
}

const deleteKnowledge = async (id: string) => {
  try {
    const response = await fetch(`/api/knowledge/${id}`, {
      method: 'DELETE'
    })
    const data = await response.json()
    if (data.status === 'success') {
      knowledgeList.value = knowledgeList.value.filter(item => item.id !== id)
    }
  } catch (error) {
    console.error('删除知识库条目失败：', error)
  }
}

const handleLogout = async () => {
  try {
    await fetch('/api/logout', { method: 'POST' })
    router.push('/login')
  } catch (error) {
    console.error('注销失败：', error)
  }
}

// 初始化加载
fetchKnowledge()
fetchHistory()
</script>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.question-box {
  max-width: 600px;
  margin: 0 auto;
}

#answer {
  background-color: #ffffff;
  padding: 15px;
  border-radius: 5px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.doc-preview {
  max-height: 300px;
  overflow-y: auto;
  background: #f8f9fa;
  padding: 15px;
  border-radius: 5px;
}

.history-box {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.list-group-item {
  transition: transform 0.2s;
}

.list-group-item:hover {
  transform: translateX(5px);
}

.upload-status {
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 10px;
}

.upload-status.success {
  background-color: #d4edda;
  color: #155724;
}

.upload-status.error {
  background-color: #f8d7da;
  color: #721c24;
}
</style>
