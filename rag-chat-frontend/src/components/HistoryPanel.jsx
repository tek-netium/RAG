import { useState } from 'react';
import { fetchHistory, clearHistory } from '../api';

export default function HistoryPanel({ sessionId }) {
    const [history, setHistory] = useState([]);
    const [show, setShow] = useState(false);

    const loadHistory = async () => {
        try {
            const data = await fetchHistory(sessionId);
            setHistory(data.history);
            setShow(true);
        } catch (err) {
            alert('获取历史失败');
        }
    };

    const handleClear = async () => {
        try {
            await clearHistory(sessionId);
            setHistory([]);
        } catch (err) {
            alert('清空失败');
        }
    };

    return (
        <div className="history-panel">
            <button className="btn" onClick={loadHistory}>历史</button>
            <button className="btn btn-danger" onClick={handleClear}>清空对话</button>
            {show && (
                <div className="history-list">
                    {history.map((msg, i) => (
                        <div key={i} className={`history-msg ${msg.role}`}>
                            <b>{msg.role === 'user' ? '你' : '助手'}: </b>
                            {msg.content}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}