<template>
  <div class="container">
    <div class="login-box">
      <h2>校园智能问答系统</h2>
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <label for="studentId">学号</label>
          <input type="text" id="studentId" v-model="studentId" required>
        </div>
        <div class="form-group">
          <label for="password">密码</label>
          <input type="password" id="password" v-model="password" required>
        </div>
        <button type="submit">登录</button>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import '../assets/styles/login.css'

const router = useRouter()
const studentId = ref('')
const password = ref('')

const handleLogin = async () => {
  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username: studentId.value,
        password: password.value
      })
    })

    const data = await response.json()
    if (data.status === 'success') {
      router.push('/')
    } else {
      alert(data.message || '登录失败')
    }
  } catch (error) {
    alert('登录失败，请稍后重试')
  }
}
</script>

<style scoped>
.container {
  max-width: 400px;
  margin: 100px auto;
  padding: 20px;
  background-color: #ffffff;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
}

.login-box {
  text-align: center;
}

.form-group {
  margin-bottom: 20px;
  text-align: left;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

input {
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ced4da;
  border-radius: 4px;
}

button {
  width: 100%;
  padding: 10px 20px;
  font-size: 16px;
  border: none;
  border-radius: 4px;
  background-color: #28a745;
  color: #ffffff;
  cursor: pointer;
}

button:hover {
  background-color: #218838;
}
</style> 