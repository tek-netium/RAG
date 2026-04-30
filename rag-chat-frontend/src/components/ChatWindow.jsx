import { useState, useEffect, useRef } from 'react';
import { streamChat } from '../api';

function formatThinkingTime(ms) {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    if (ms < 10000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 1000).toFixed(0)}s`;
}

export default function ChatWindow({ sessionId }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const thinkingStartRef = useRef(null);
    const thinkingTimerRef = useRef(null);
    const firstEventRef = useRef(false);
    const [thinkingDisplay, setThinkingDisplay] = useState(null);
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, thinkingDisplay]);

    const clearThinkingTimer = () => {
        if (thinkingTimerRef.current) {
            clearInterval(thinkingTimerRef.current);
            thinkingTimerRef.current = null;
        }
    };

    useEffect(() => {
        return () => clearThinkingTimer();
    }, []);

    const handleSend = async () => {
        if (!input.trim() || loading) return;
        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        thinkingStartRef.current = Date.now();
        firstEventRef.current = false;
        setThinkingDisplay('思考中…');

        thinkingTimerRef.current = setInterval(() => {
            const elapsed = Date.now() - thinkingStartRef.current;
            setThinkingDisplay(`思考中… ${formatThinkingTime(elapsed)}`);
        }, 100);

        const assistantMessage = { role: 'assistant', content: '', retrieved: [], thinkingTimeMs: null };
        setMessages(prev => [...prev, assistantMessage]);

        const onFirstEvent = () => {
            if (!firstEventRef.current) {
                firstEventRef.current = true;
                clearThinkingTimer();
                const finalTime = Date.now() - thinkingStartRef.current;
                setThinkingDisplay(null);
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    return [...prev.slice(0, -1), { ...last, thinkingTimeMs: finalTime }];
                });
            }
        };

        try {
            for await (const event of streamChat({ message: input, sessionId })) {
                onFirstEvent();
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
            clearThinkingTimer();
            setThinkingDisplay(null);
            setMessages(prev => [
                ...prev.slice(0, -1),
                { ...prev[prev.length - 1], content: '❌ 请求失败' }
            ]);
        } finally {
            clearThinkingTimer();
            setThinkingDisplay(null);
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
                            {msg.thinkingTimeMs != null && (
                                <div className="thinking-time">
                                    思考用时 {formatThinkingTime(msg.thinkingTimeMs)}
                                </div>
                            )}
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
                {thinkingDisplay && (
                    <div className="thinking-indicator">{thinkingDisplay}</div>
                )}
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