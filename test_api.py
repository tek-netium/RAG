import requests
import json
import time

BASE_URL = "http://localhost:8000"

# 1. 上传文档
file_path = "D:/cursor_projects/RAG/test.md"  # 改为你的文件路径
with open(file_path, "rb") as f:
    resp = requests.post(f"{BASE_URL}/documents/upload", files={"file": f})
print("Upload response:", resp.json())

# 2. 等待后台处理完成（简单等待 5 秒，实际可以轮询 /documents 确认）
print("等待文档索引...")
time.sleep(5)

# 3. 发送聊天请求（流式）
chat_payload = {"message": "总结一下这个文档的主要内容", "session_id": "python_test"}
resp = requests.post(f"{BASE_URL}/chat", json=chat_payload, stream=True)
print("Chat response (SSE stream):")
for line in resp.iter_lines():
    if line:
        print(line.decode("utf-8"))

# 4. 获取历史
hist_resp = requests.get(f"{BASE_URL}/history/python_test")
print("\nHistory:", hist_resp.json())

# 5. 清除历史
clear_resp = requests.delete(f"{BASE_URL}/history/python_test")
print("Clear history:", clear_resp.json())

# 6. 列出文档
docs_resp = requests.get(f"{BASE_URL}/documents")
print("Documents:", docs_resp.json())
