import { useState, useEffect, useRef } from 'react';
import { listDocuments, uploadDocument, deleteDocument } from '../api';

export default function DocumentManager() {
    const [docs, setDocs] = useState([]);
    const fileInputRef = useRef(null);

    const loadDocs = async () => {
        try {
            const res = await listDocuments();
            setDocs(res.documents);
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        loadDocs();
    }, []);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            await uploadDocument(file);
            loadDocs();
        } catch (err) {
            alert('上传失败');
        }
    };

    const handleDelete = async (filename) => {
        try {
            await deleteDocument(filename);
            loadDocs();
        } catch (err) {
            alert('删除失败');
        }
    };

    return (
        <div className="document-manager">
            <h3>知识库文档</h3>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleUpload}
                style={{ display: 'none' }}
            />
            <div className="doc-actions">
                <button className="btn btn-primary" onClick={() => fileInputRef.current.click()}>上传文档</button>
            </div>
            <ul>
                {docs.map(doc => (
                    <li key={doc}>
                        <span className="doc-name" title={doc}>{doc}</span>
                        <button className="btn btn-danger" onClick={() => handleDelete(doc)}>删除</button>
                    </li>
                ))}
            </ul>
        </div>
    );
}