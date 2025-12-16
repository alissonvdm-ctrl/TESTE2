'use client';

import { FormEvent, useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';

type ParentOption = {
  id: string;
  title: string;
};

type AttachmentInput = {
  name: string;
  url: string;
  mimeType: string;
  size?: string;
};

type FAQFormProps = {
  parentOptions?: ParentOption[];
};

export default function FAQForm({ parentOptions = [] }: FAQFormProps) {
  const [title, setTitle] = useState('');
  const [summary, setSummary] = useState('');
  const [content, setContent] = useState('');
  const [topics, setTopics] = useState('');
  const [status, setStatus] = useState<'DRAFT' | 'PUBLISHED'>('DRAFT');
  const [parentId, setParentId] = useState('');
  const [authorEmail, setAuthorEmail] = useState('autor.demo@faq.local');
  const [attachments, setAttachments] = useState<AttachmentInput[]>([]);
  const [shareEmails, setShareEmails] = useState('');
  const [sharePermission, setSharePermission] = useState<'VIEW' | 'EDIT' | 'ADMIN'>('VIEW');
  const [message, setMessage] = useState('');
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  function addAttachment() {
    setAttachments((prev) => [...prev, { name: '', url: '', mimeType: 'application/octet-stream' }]);
  }

  function updateAttachment(index: number, field: keyof AttachmentInput, value: string) {
    setAttachments((prev) => {
      const copy = [...prev];
      copy[index] = { ...copy[index], [field]: value };
      return copy;
    });
  }

  function removeAttachment(index: number) {
    setAttachments((prev) => prev.filter((_, idx) => idx !== index));
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setMessage('');

    const parsedAttachments = attachments
      .filter((att) => att.name && att.url)
      .map((att) => ({
        ...att,
        size: att.size ? Number(att.size) : undefined
      }));

    const body = {
      title,
      summary,
      content,
      status,
      parentId: parentId || null,
      topicNames: topics.split(',').map((t) => t.trim()).filter(Boolean),
      authorEmail,
      attachments: parsedAttachments,
      shares: shareEmails
        .split(',')
        .map((email) => email.trim())
        .filter(Boolean)
        .map((email) => ({ email, permission: sharePermission }))
    };

    const res = await fetch('/api/faqs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (res.ok) {
      setMessage('FAQ salva com sucesso.');
      setTitle('');
      setSummary('');
      setContent('');
      setTopics('');
      setStatus('DRAFT');
      setParentId('');
      setAuthorEmail('autor.demo@faq.local');
      setAttachments([]);
      setShareEmails('');
      startTransition(() => router.refresh());
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

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
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

        <label className="flex flex-col gap-1 text-sm text-slate-200">
          FAQ pai (opcional)
          <select
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={parentId}
            onChange={(e) => setParentId(e.target.value)}
          >
            <option value="">Nenhuma</option>
            {parentOptions.map((parent) => (
              <option key={parent.id} value={parent.id}>
                {parent.title}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          E-mail do autor
          <input
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            value={authorEmail}
            onChange={(e) => setAuthorEmail(e.target.value)}
            type="email"
            required
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Compartilhar com (emails separados por vírgula)
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
              value={shareEmails}
              onChange={(e) => setShareEmails(e.target.value)}
              placeholder="ex: pessoa@empresa.com, outro@dominio.com"
            />
            <select
              className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
              value={sharePermission}
              onChange={(e) => setSharePermission(e.target.value as 'VIEW' | 'EDIT' | 'ADMIN')}
            >
              <option value="VIEW">Visualizar</option>
              <option value="EDIT">Editar</option>
              <option value="ADMIN">Admin</option>
            </select>
          </div>
        </label>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-slate-200">
          <span>Anexos (URL externa ou bucket)</span>
          <button
            type="button"
            onClick={addAttachment}
            className="text-emerald-300 hover:text-emerald-200"
          >
            + Adicionar anexo
          </button>
        </div>

        {attachments.length === 0 && (
          <p className="text-xs text-slate-400">Nenhum anexo. Clique em "Adicionar anexo".</p>
        )}

        <div className="space-y-3">
          {attachments.map((attachment, index) => (
            <div
              key={`${attachment.name}-${index}`}
              className="grid grid-cols-1 gap-2 rounded-lg border border-slate-800 bg-slate-950/50 p-3 md:grid-cols-4"
            >
              <input
                className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
                placeholder="Nome do arquivo"
                value={attachment.name}
                onChange={(e) => updateAttachment(index, 'name', e.target.value)}
              />
              <input
                className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100 md:col-span-2"
                placeholder="URL do anexo"
                value={attachment.url}
                onChange={(e) => updateAttachment(index, 'url', e.target.value)}
              />
              <div className="flex gap-2">
                <input
                  className="w-32 rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
                  placeholder="MIME"
                  value={attachment.mimeType}
                  onChange={(e) => updateAttachment(index, 'mimeType', e.target.value)}
                />
                <input
                  className="w-28 rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
                  placeholder="Tamanho"
                  type="number"
                  value={attachment.size ?? ''}
                  onChange={(e) => updateAttachment(index, 'size', e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => removeAttachment(index)}
                  className="rounded-md border border-red-400 px-2 py-1 text-red-200 hover:bg-red-500/10"
                >
                  Remover
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={isPending}
          className="rounded-lg bg-emerald-500 px-4 py-2 text-slate-900 font-semibold hover:bg-emerald-400 disabled:opacity-70"
        >
          {isPending ? 'Salvando...' : 'Salvar FAQ'}
        </button>
        {message && <p className="text-sm text-emerald-200">{message}</p>}
      </div>
    </form>
  );
}
