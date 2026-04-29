import { useState, useEffect, useRef } from 'react';
import { streamChat } from '../api';

export default function ChatWindow({ sessionId }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;
        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        // 为助手回复占位
        const assistantMessage = { role: 'assistant', content: '', retrieved: [] };
        setMessages(prev => [...prev, assistantMessage]);

        try {
            for await (const event of streamChat({ message: input, sessionId })) {
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    if (event.type === 'answer_delta') {
                        return [
                            ...prev.slice(0, -1),
                            { ...last, content: last.content + event.content }
                        ];
                    } else if (event.type === 'retrieved_context') {
                        return [
                            ...prev.slice(0, -1),
                            { ...last, retrieved: [...last.retrieved, event.content] }
                        ];
                    } else if (event.type === 'error') {
                        return [
                            ...prev.slice(0, -1),
                            { ...last, content: `❌ 错误: ${event.content}` }
                        ];
                    }
                    return prev;
                });
            }
        } catch (err) {
            setMessages(prev => [
                ...prev.slice(0, -1),
                { ...prev[prev.length - 1], content: '❌ 请求失败' }
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, i) => (
                    <div key={i} className={`message ${msg.role}`}>
                        <div className="role">{msg.role === 'user' ? '你' : '助手'}</div>
                        <div className="bubble">
                            <div className="content">{msg.content}</div>
                        {msg.retrieved?.length > 0 && (
                            <details>
                                <summary>检索上下文</summary>
                                {msg.retrieved.map((ctx, idx) => (
                                    <p key={idx} className="context">{ctx}</p>
                                ))}
                            </details>
                        )}
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
            <div className="input-area">
                <div className="composer">
                    <input
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSend()}
                        placeholder="输入问题…"
                        disabled={loading}
                    />
                    <div className="composer-actions">
                        <button className="btn btn-primary" onClick={handleSend} disabled={loading}>
                            {loading ? '发送中…' : '发送'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}