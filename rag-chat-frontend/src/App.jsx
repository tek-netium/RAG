import { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import HistoryPanel from './components/HistoryPanel';
import DocumentManager from './components/DocumentManager';
import './App.css';

function App() {
  const [sessionId] = useState('default_web_session');

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-title">RAG 智能问答</div>
          <div className="brand-subtitle">检索增强生成 · 知识库问答</div>
        </div>
        <div className="topbar-actions">
          <HistoryPanel sessionId={sessionId} />
        </div>
      </header>
      <main className="layout">
        <section className="chat-section">
          <ChatWindow sessionId={sessionId} />
        </section>
        <aside className="doc-section">
          <DocumentManager />
        </aside>
      </main>
    </div>
  );
}

export default App;
