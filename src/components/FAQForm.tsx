'use client';

import { useState } from 'react';

export default function FAQForm() {
  const [title, setTitle] = useState('');
  const [summary, setSummary] = useState('');
  const [content, setContent] = useState('');
  const [topics, setTopics] = useState('');
  const [status, setStatus] = useState<'DRAFT' | 'PUBLISHED'>('DRAFT');
  const [message, setMessage] = useState('');

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setMessage('');

    const body = {
      title,
      summary,
      content,
      status,
      topicNames: topics.split(',').map((t) => t.trim()).filter(Boolean),
      authorId: 'demo-author'
    };

    const res = await fetch('/api/faqs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (res.ok) {
      setMessage('FAQ salva com sucesso. Atualize a página para ver na lista.');
      setTitle('');
      setSummary('');
      setContent('');
      setTopics('');
      setStatus('DRAFT');
    } else {
      setMessage('Erro ao salvar FAQ. Verifique o console.');
      console.error(await res.text());
    }
  }

  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Título
          <input
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Resumo
          <input
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={summary}
            onChange={(e) => setSummary(e.target.value)}
            placeholder="Descrição breve"
          />
        </label>
      </div>
      <label className="flex flex-col gap-1 text-sm text-slate-200">
        Conteúdo
        <textarea
          className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 min-h-[120px]"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          required
        />
      </label>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Assuntos (separados por vírgula)
          <input
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={topics}
            onChange={(e) => setTopics(e.target.value)}
            placeholder="Ex: Onboarding, Segurança"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Status
          <select
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={status}
            onChange={(e) => setStatus(e.target.value as 'DRAFT' | 'PUBLISHED')}
          >
            <option value="DRAFT">Rascunho</option>
            <option value="PUBLISHED">Publicado</option>
          </select>
        </label>
      </div>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          className="rounded-lg bg-emerald-500 px-4 py-2 text-slate-900 font-semibold hover:bg-emerald-400"
        >
          Salvar FAQ
        </button>
        {message && <p className="text-sm text-emerald-200">{message}</p>}
      </div>
    </form>
  );
}
