const BASE_URL = import.meta.env.VITE_API_BASE || '';

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
export async function uploadDocument(file, sessionId) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/documents/upload?session_id=${encodeURIComponent(sessionId)}`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error('上传失败');
    return res.json();
}

// 获取所有已上传文档
export async function listDocuments(sessionId) {
    const res = await fetch(`${BASE_URL}/documents?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error('获取文档列表失败');
    return res.json();
}

// 删除指定文档
export async function deleteDocument(filename, sessionId) {
    const res = await fetch(`${BASE_URL}/documents/${filename}?session_id=${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('删除失败');
    return res.json();
}

// 获取所有会话
export async function listSessions() {
    const res = await fetch(`${BASE_URL}/sessions`);
    if (!res.ok) throw new Error('获取会话列表失败');
    return res.json();
}

// 创建新会话
export async function createSession(name) {
    const res = await fetch(`${BASE_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error('创建会话失败');
    return res.json();
}

// 删除会话
export async function deleteSession(sessionId) {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('删除会话失败');
    return res.json();
}