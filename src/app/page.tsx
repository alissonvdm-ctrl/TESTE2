import Link from 'next/link';
import FAQForm from '@/components/FAQForm';
import { prisma } from '@/lib/prisma';

async function getFaqs() {
  const faqs = await prisma.fAQ.findMany({
    include: { topics: { include: { topic: true } }, attachments: true },
    orderBy: { updatedAt: 'desc' }
  });

  return faqs.map((faq) => ({
    ...faq,
    topics: faq.topics.map((t) => t.topic)
  }));
}

export default async function Home() {
  const faqs = await getFaqs();

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <div>
          <p className="text-sm text-slate-400">Envie novas perguntas ou edite conteúdos existentes</p>
          <h2 className="text-xl font-semibold text-white">Criar FAQ</h2>
        </div>
        <FAQForm />
      </section>

      <section className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">FAQs recentes</h2>
          <Link className="text-sm text-emerald-300" href="/api/faqs" prefetch={false}>
            API aberta
          </Link>
        </div>
        <div className="space-y-3">
          {faqs.map((faq) => (
            <article key={faq.id} className="p-4 rounded-xl border border-slate-800 bg-slate-900/50">
              <header className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">{faq.status}</p>
                  <h3 className="text-lg font-semibold">{faq.title}</h3>
                  <p className="text-sm text-slate-300">{faq.summary ?? 'Sem resumo'}</p>
                </div>
                <span className="badge">{faq.topics.map((t) => t.name).join(', ') || 'Sem assuntos'}</span>
              </header>
              <p className="text-sm text-slate-200 mt-2">{faq.content}</p>
              <footer className="text-xs text-slate-500 mt-3 flex justify-between">
                <span>Prioridade: {faq.priority}</span>
                <span>Atualizado em {new Date(faq.updatedAt).toLocaleDateString('pt-BR')}</span>
              </footer>
            </article>
          ))}
          {faqs.length === 0 && <p className="text-slate-400">Nenhuma FAQ cadastrada ainda.</p>}
        </div>
      </section>
    </main>
  );
}
