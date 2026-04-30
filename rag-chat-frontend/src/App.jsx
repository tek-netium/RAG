import { useState, useEffect, useCallback } from 'react';
import ChatWindow from './components/ChatWindow';
import HistoryPanel from './components/HistoryPanel';
import DocumentManager from './components/DocumentManager';
import { listSessions, createSession, deleteSession } from './api';
import './App.css';

function App() {
  const [sessionId, setSessionId] = useState('default');
  const [sessions, setSessions] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const loadSessions = useCallback(async () => {
    try {
      const res = await listSessions();
      setSessions(res.sessions || []);
    } catch {
      // backend may not have sessions yet
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleCreate = async () => {
    try {
      const res = await createSession('');
      setSessions(prev => [...prev, res]);
      setSessionId(res.session_id);
    } catch {
      alert('创建会话失败');
    }
  };

  const handleSwitch = (sid) => {
    setSessionId(sid);
    setSidebarOpen(false);
  };

  const handleDelete = async (sid) => {
    if (!confirm('确定要删除该会话及其所有文档和聊天记录吗？')) return;
    try {
      await deleteSession(sid);
      setSessions(prev => prev.filter(s => s.session_id !== sid));
      if (sid === sessionId) {
        setSessionId('default');
      }
    } catch {
      alert('删除会话失败');
    }
  };

  const currentName = sessions.find(s => s.session_id === sessionId)?.name || sessionId.slice(0, 8);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-left">
          <button className="btn sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? '✕' : '☰'}
          </button>
          <div className="brand">
            <div className="brand-title">RAG 智能问答</div>
            <div className="brand-subtitle">检索增强生成 · 知识库问答</div>
          </div>
        </div>
        <div className="topbar-actions">
          <HistoryPanel sessionId={sessionId} />
        </div>
      </header>

      <div className={`app-body ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <aside className="session-sidebar">
          <div className="sidebar-header">
            <span className="sidebar-title">会话列表</span>
            <button className="btn btn-primary btn-sm" onClick={handleCreate}>+ 新建</button>
          </div>
          <ul className="session-list">
            {sessions.map(s => (
              <li
                key={s.session_id}
                className={`session-item ${s.session_id === sessionId ? 'active' : ''}`}
                onClick={() => handleSwitch(s.session_id)}
              >
                <span className="session-name">{s.name}</span>
                <button
                  className="btn btn-danger btn-xs"
                  onClick={(e) => { e.stopPropagation(); handleDelete(s.session_id); }}
                >
                  删除
                </button>
              </li>
            ))}
            {sessions.length === 0 && (
              <li className="session-empty">暂无会话，点击"+ 新建"创建</li>
            )}
          </ul>
        </aside>

        <main className="layout">
          <section className="chat-section">
            <ChatWindow sessionId={sessionId} />
          </section>
          <aside className="doc-section">
            <DocumentManager sessionId={sessionId} />
          </aside>
        </main>
      </div>
    </div>
  );
}

export default App;
