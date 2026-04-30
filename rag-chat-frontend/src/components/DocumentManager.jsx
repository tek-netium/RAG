import { useState, useEffect, useRef } from 'react';
import { listDocuments, uploadDocument, deleteDocument } from '../api';

export default function DocumentManager({ sessionId }) {
    const [docs, setDocs] = useState([]);
    const [uploading, setUploading] = useState(new Set());
    const [uploadProgress, setUploadProgress] = useState(null);
    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);

    const loadDocs = async () => {
        try {
            const res = await listDocuments(sessionId);
            setDocs(res.documents);
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        setDocs([]);
        loadDocs();
    }, [sessionId]);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        await uploadOne(file);
        e.target.value = '';
    };

    const handleUploadFolder = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        setUploadProgress({ current: 0, total: files.length });
        for (let i = 0; i < files.length; i++) {
            await uploadOne(files[i]);
            setUploadProgress({ current: i + 1, total: files.length });
        }
        setUploadProgress(null);
        e.target.value = '';
    };

    const uploadOne = async (file) => {
        const filename = file.name;
        try {
            setUploading(prev => new Set(prev).add(filename));
            await uploadDocument(file, sessionId);
            setDocs(prev => {
                if (!prev.includes(filename)) return [...prev, filename];
                return prev;
            });
            pollUntilIndexed(filename);
        } catch (err) {
            setUploading(prev => {
                const next = new Set(prev);
                next.delete(filename);
                return next;
            });
            console.error(`上传失败: ${filename}`, err);
        }
    };

    const pollUntilIndexed = (filename) => {
        let attempts = 0;
        const interval = setInterval(async () => {
            attempts++;
            try {
                const res = await listDocuments(sessionId);
                if (res.documents.includes(filename)) {
                    clearInterval(interval);
                    setDocs(res.documents);
                    setUploading(prev => {
                        const next = new Set(prev);
                        next.delete(filename);
                        return next;
                    });
                } else if (attempts >= 6) {
                    clearInterval(interval);
                    loadDocs();
                    setUploading(prev => {
                        const next = new Set(prev);
                        next.delete(filename);
                        return next;
                    });
                }
            } catch {
                if (attempts >= 6) {
                    clearInterval(interval);
                    loadDocs();
                    setUploading(prev => {
                        const next = new Set(prev);
                        next.delete(filename);
                        return next;
                    });
                }
            }
        }, 800);
    };

    const handleDelete = async (filename) => {
        setDocs(prev => prev.filter(d => d !== filename));
        try {
            await deleteDocument(filename, sessionId);
        } catch (err) {
            loadDocs();
        }
    };

    const isUploading = uploading.size > 0;

    return (
        <div className="document-manager">
            <h3>知识库文档</h3>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleUpload}
                style={{ display: 'none' }}
            />
            <input
                type="file"
                ref={folderInputRef}
                webkitdirectory=""
                onChange={handleUploadFolder}
                style={{ display: 'none' }}
            />
            <div className="doc-actions">
                <button className="btn btn-primary" onClick={() => fileInputRef.current.click()}>上传文档</button>
                <button className="btn btn-primary" onClick={() => folderInputRef.current.click()}>上传文件夹</button>
            </div>
            {uploadProgress && (
                <div className="upload-progress">
                    上传中 {uploadProgress.current}/{uploadProgress.total}
                </div>
            )}
            <ul>
                {docs.map(doc => (
                    <li key={doc} className={uploading.has(doc) ? 'doc-uploading' : ''}>
                        <span className="doc-name" title={doc}>{doc}</span>
                        {uploading.has(doc) ? (
                            <span className="doc-status">处理中…</span>
                        ) : (
                            <button className="btn btn-danger" onClick={() => handleDelete(doc)}>删除</button>
                        )}
                    </li>
                ))}
                {docs.length === 0 && !isUploading && (
                    <li className="doc-empty">暂无文档，上传文档开始构建知识库</li>
                )}
            </ul>
        </div>
    );
}