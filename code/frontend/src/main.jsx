import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

async function api(path, options = {}) {
  const response = await fetch(`/api/${path}`, {...options, signal: AbortSignal.timeout(120000)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.data?.message || 'Không thể kết nối API.');
  return body;
}

function Metrics({data}) {
  const fmt = value => value == null ? 'N/A' : Number(value).toFixed(3);
  const models = {...data.models, ...(data.baselines?.M1 ? {'CLIP cosine (M1)': data.baselines.M1} : {})};
  const scope = data.scope === 'experimental_ensemble_validation' ? 'Ensemble thử nghiệm: bảng validation; mean/ridge/mlp là đối chứng cũ. Chưa nghiệm thu release.' : data.scope === 'reviewed_full_data_validation_only' ? 'GenAI-Bench: số liệu validation. Kết quả test holdout được báo cáo riêng.' : data.scope === 'development_only_not_dataset_benchmark' ? 'Dữ liệu thử nghiệm — không phải benchmark GenAI-Bench.' : data.scope;
  return <section className="metrics"><h3>Kết quả validation</h3><p>{scope}</p>
    <div className="table-scroll"><table><thead><tr><th scope="col">Mô hình</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">Spearman</th><th scope="col">Pairwise accuracy</th></tr></thead><tbody>
      {Object.entries(models).map(([name, values]) => <tr key={name}><td>{name}</td><td>{fmt(values.mae)}</td><td>{fmt(values.rmse)}</td><td>{fmt(values.spearman)}</td><td>{fmt(values.pairwise?.accuracy)}</td></tr>)}
    </tbody></table></div><small>Split: {data.split} · N/A: không xác định hoặc không áp dụng. Cosine M1 không phải điểm 1–5.</small></section>;
}

function App() {
  const [health, setHealth] = useState(null), [session, setSession] = useState(null);
  const [prompt, setPrompt] = useState(''), [files, setFiles] = useState([]), [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [text, setText] = useState('');
  const previews = useRef([]), bottom = useRef(null);
  useEffect(() => {api('health').then(r => setHealth(r.data)).catch(e => setError(e.message)); return () => previews.current.forEach(URL.revokeObjectURL);}, []);
  useEffect(() => {bottom.current?.scrollIntoView({behavior: 'smooth'});}, [messages]);

  function selectFiles(event) {
    const selected = Array.from(event.target.files || []);
    if (selected.length + files.length > 6) {setError('Chọn tối đa 6 ảnh.'); return;}
    if (selected.some(f => !['image/png', 'image/jpeg', 'image/webp'].includes(f.type) || f.size > 10 * 1024 * 1024)) {
      setError('Ảnh cần là JPG, PNG hoặc WebP, tối đa 10 MB mỗi ảnh.'); return;
    }
    const added = selected.map(file => ({file, url: URL.createObjectURL(file)}));
    previews.current.push(...added.map(f => f.url)); setFiles([...files, ...added]); setError(''); event.target.value = '';
  }

  async function submit(action, followup = '') {
    if (busy) return;
    if (['evaluate', 'compare'].includes(action) && (!prompt.trim() || (action === 'evaluate' ? files.length !== 1 : files.length < 2))) {
      setError('Nhập prompt và chọn 1 ảnh để chấm hoặc 2–6 ảnh để so sánh.'); return;
    }
    setBusy(true); setError('');
    try {
      let reply;
      if (action === 'metrics') reply = await api('metrics');
      else {
        let id = session;
        if (!id) {id = (await api('sessions', {method: 'POST'})).data.session_id; setSession(id);}
        let imageIds = [];
        if (action !== 'auto' && files.length) {
          const form = new FormData(); form.append('session_id', id); files.forEach(f => form.append('files', f.file));
          imageIds = (await api('uploads', {method: 'POST', body: form})).data.image_ids;
        }
        reply = await api('chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({session_id: id, action,
          prompt: action === 'auto' ? null : prompt, image_ids: imageIds, text: followup})});
      }
      setMessages(old => [...old, {role: 'user', text: followup || (action === 'metrics' ? 'Xem kết quả đánh giá' : prompt)},
        {role: 'assistant', reply, thumbnails: action === 'auto' ? [] : files.map(f => f.url)}]); setText('');
    } catch (e) {setError(e.name === 'TimeoutError' ? 'Yêu cầu quá thời gian. Hãy thử lại.' : e.message);}
    finally {setBusy(false);}
  }

  return <div className="layout"><aside><a className="brand" href="/">◈ <span>PromptAlign<span className="brand-ai"> AI</span></span></a>
    <p className="eyebrow">KHÔNG GIAN ĐÁNH GIÁ</p><button className="nav active" onClick={() => document.getElementById('prompt').focus()}>◉ Hội thoại & ảnh</button>
    <button className="nav" disabled={busy} onClick={() => submit('metrics')}>▤ Kết quả đánh giá</button>
    <div className="sidebar-note"><span className="dot"/> {health?.model_ready ? 'Mô hình thử nghiệm đã kết nối' : 'Chờ mô hình được cấu hình'}<p>Điểm phù hợp 1–5 không phải xác suất hay phần trăm chính xác.</p></div>
  </aside><main><header><span>WORKSPACE / PROMPT–IMAGE</span><button onClick={() => {setSession(null); setMessages([]); setError('');}}>＋ Phiên mới</button></header>
    <section className="intro"><p className="eyebrow">HIỂU SỰ PHÙ HỢP GIỮA LỜI VÀ ẢNH</p><h1>Từ mô tả đến<br/><em>mức độ phù hợp.</em></h1><p>Đưa vào một prompt và những hình ảnh bạn muốn đánh giá.<br/>Chấm điểm, so sánh và xem kết quả trong cùng một cuộc trò chuyện.</p>
    <div className="suggestions">{['Chấm một ảnh', 'So sánh nhiều ảnh', 'Xem biểu đồ', 'Cách sử dụng'].map((label, i) => <button key={label} disabled={busy} onClick={() => i === 2 ? submit('metrics') : i === 3 ? submit('auto', 'hướng dẫn') : document.getElementById('prompt').focus()}>{label} ↗</button>)}</div></section>
    {health && !health.model_ready && <p className="notice">Chưa có mô hình sẵn sàng. Bạn có thể chuẩn bị ảnh và prompt; hệ thống sẽ không tạo điểm giả.</p>}
    {health?.model_ready && <p className="notice">{health.scope === 'experimental_ensemble_validation' ? 'Đang dùng ensemble thử nghiệm. Còn hạn chế đếm và trạng thái; chưa nghiệm thu release.' : health.scope === 'reviewed_full_data_validation_only' ? 'Mô hình đã kiểm tra trên validation GenAI-Bench. Bảng metrics hiển thị validation; chưa nghiệm thu release.' : 'Chế độ thử nghiệm local. Checkpoint chưa được nghiệm thu trên GenAI-Bench.'}</p>}
    <section className="conversation" aria-live="polite">{messages.map((message, i) => <article key={i} className={`message ${message.role}`}>
      <span className="speaker">{message.role === 'user' ? 'BẠN' : 'PROMPTALIGN'}</span>
      {message.text && <p>{message.text}</p>}
      {message.reply?.type === 'text' && <p>{message.reply.data.text}</p>}
      {message.reply?.type === 'metrics' && <Metrics data={message.reply.data}/>}
      {message.reply?.data.items && <div className="results">{message.reply.data.items.map(item => <div className="result" key={item.index}>
        {message.thumbnails[item.index] && <img src={message.thumbnails[item.index]} alt={`Ảnh ${item.index + 1}`}/>}<span>Ảnh {item.index + 1} · Hạng {item.rank}</span><strong>{item.score.toFixed(2)}<small> / 5</small></strong><small>CLIP cosine: {item.cosine.toFixed(3)}</small>{typeof item.itm === 'number' && <small>Khớp prompt (ITM): {Math.round(item.itm * 100)}%</small>}{item.warning && <small className="warning">{item.warning}</small>}</div>)}</div>}
      {message.reply && <small className="metadata">{message.reply.model_version || 'Chưa có model'} · {message.reply.elapsed_ms} ms</small>}
    </article>)}<div ref={bottom}/></section>
    <section className="composer"><label htmlFor="prompt">Prompt dùng để đối chiếu</label><textarea id="prompt" value={prompt} maxLength={1000} onChange={e => setPrompt(e.target.value)} placeholder="Ví dụ: A red bicycle leaning against a stone wall…"/><div className="previews">{files.map((f, i) => <div key={f.url}><img src={f.url} alt={`Ảnh đã chọn ${i + 1}`}/><button aria-label={`Xóa ảnh ${i + 1}`} onClick={() => setFiles(files.filter((_, j) => j !== i))}>×</button></div>)}</div>
      <div className="actions"><label className="upload">＋ Thêm ảnh<input aria-label="Chọn ảnh" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={selectFiles}/></label><span>{files.length}/6 ảnh · {prompt.length}/1000</span><button disabled={busy} onClick={() => submit('evaluate')}>Chấm ảnh</button><button className="primary" disabled={busy} onClick={() => submit('compare')}>So sánh →</button></div>
      {messages.length > 0 && <form className="followup" onSubmit={e => {e.preventDefault(); submit('auto', text);}}><input aria-label="Hỏi tiếp" value={text} onChange={e => setText(e.target.value)} maxLength={1000} placeholder="Hỏi tiếp: ảnh thứ hai thì sao?"/><button disabled={busy || !text.trim()}>Gửi</button></form>}
    </section>{busy && <p role="status">Đang xử lý…</p>}{error && <p className="error" role="alert">{error} <button onClick={() => setError('')}>Đóng</button></p>}<footer>PromptAlign AI · CLIP + Regression · JPG / PNG / WebP, tối đa 10 MB mỗi ảnh</footer>
  </main></div>;
}

createRoot(document.getElementById('root')).render(<App/>);
