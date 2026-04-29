const BASE_URL = ''; // 使用代理，无需写全域名

// 处理 SSE 流（用于 /chat）
export async function* streamChat({ message, sessionId }) {
    const response = await fetch(`${BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        // 保留最后一个可能不完整的行到 buffer
        buffer = lines.pop();

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                yield JSON.parse(data);
            }
        }
    }
}

// 获取历史消息
export async function fetchHistory(sessionId) {
    const res = await fetch(`${BASE_URL}/history/${sessionId}`);
    if (!res.ok) throw new Error('获取历史失败');
    return res.json();
}

// 清空历史消息
export async function clearHistory(sessionId) {
    const res = await fetch(`${BASE_URL}/history/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('清空历史失败');
    return res.json();
}

// 上传文档
export async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error('上传失败');
    return res.json();
}

// 获取所有已上传文档
export async function listDocuments() {
    const res = await fetch(`${BASE_URL}/documents`);
    if (!res.ok) throw new Error('获取文档列表失败');
    return res.json();
}

// 删除指定文档
export async function deleteDocument(filename) {
    const res = await fetch(`${BASE_URL}/documents/${filename}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('删除失败');
    return res.json();
}