import { Prisma } from '@prisma/client';
import Link from 'next/link';
import FAQForm from '@/components/FAQForm';
import FilterBar from '@/components/FilterBar';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';

type SearchParams = {
  q?: string;
  topics?: string;
};

function parseTopicFilters(topics?: string) {
  if (!topics) return [] as string[];
  return topics
    .split(',')
    .map((topic) => topic.trim())
    .filter(Boolean);
}

async function getFaqs({ q, topicFilters }: { q?: string; topicFilters: string[] }) {
  const where: Prisma.FAQWhereInput = {
    ...(q
      ? {
          OR: [
            { title: { contains: q, mode: 'insensitive' } },
            { summary: { contains: q, mode: 'insensitive' } },
            { content: { contains: q, mode: 'insensitive' } }
          ]
        }
      : {}),
    ...(topicFilters.length
      ? { topics: { some: { topic: { name: { in: topicFilters, mode: 'insensitive' } } } } }
      : {})
  };

  const faqs = await prisma.fAQ.findMany({
    where,
    include: {
      topics: { include: { topic: true } },
      attachments: true,
      shares: { include: { user: true } },
      parent: { select: { id: true, title: true } },
      _count: { select: { children: true } }
    },
    orderBy: { updatedAt: 'desc' }
  });

  return faqs.map((faq) => ({
    ...faq,
    topics: faq.topics.map((t) => t.topic)
  }));
}

async function getTopics() {
  return prisma.topic.findMany({
    include: { _count: { select: { faqs: true } } },
    orderBy: { name: 'asc' }
  });
}

async function getParentOptions() {
  return prisma.fAQ.findMany({
    select: { id: true, title: true },
    orderBy: { updatedAt: 'desc' }
  });
}

export default async function Home({ searchParams }: { searchParams?: SearchParams }) {
  const q = searchParams?.q ?? '';
  const topicFilters = parseTopicFilters(searchParams?.topics);

  const [faqs, topics, parentOptions] = await Promise.all([
    getFaqs({ q, topicFilters }),
    getTopics(),
    getParentOptions()
  ]);

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <div>
          <p className="text-sm text-slate-400">Envie novas perguntas ou edite conteúdos existentes</p>
          <h2 className="text-xl font-semibold text-white">Criar FAQ</h2>
        </div>
        <FAQForm parentOptions={parentOptions} />
      </section>

      <section className="card space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">Pesquisar, filtrar por assunto e explorar hierarquias</p>
            <h2 className="text-lg font-semibold text-white">FAQs recentes</h2>
          </div>
          <Link className="text-sm text-emerald-300" href="/api/faqs" prefetch={false}>
            API aberta
          </Link>
        </div>

        <FilterBar topics={topics} initialSearch={q} selectedTopics={topicFilters} />

        <div className="space-y-3">
          {faqs.map((faq) => (
            <article key={faq.id} className="p-4 rounded-xl border border-slate-800 bg-slate-900/50 space-y-2">
              <header className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between md:gap-2">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">{faq.status}</p>
                  <h3 className="text-lg font-semibold">{faq.title}</h3>
                  <p className="text-sm text-slate-300">{faq.summary ?? 'Sem resumo'}</p>
                  {faq.parent && (
                    <p className="text-xs text-slate-400">Subartigo de: {faq.parent.title}</p>
                  )}
                </div>
                <span className="badge">{faq.topics.map((t) => t.name).join(', ') || 'Sem assuntos'}</span>
              </header>

              <p className="text-sm text-slate-200 whitespace-pre-wrap">{faq.content}</p>

              <div className="flex flex-wrap gap-2 text-xs text-slate-300">
                <span className="badge">Prioridade: {faq.priority}</span>
                <span className="badge">Atualizado em {new Date(faq.updatedAt).toLocaleDateString('pt-BR')}</span>
                <span className="badge">Sub-faqs: {faq._count.children}</span>
                <span className="badge">Compartilhamentos: {faq.shares.length}</span>
              </div>

              {faq.attachments.length > 0 && (
                <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 space-y-1">
                  <p className="text-xs font-semibold text-slate-300">Anexos ({faq.attachments.length})</p>
                  <ul className="text-sm space-y-1">
                    {faq.attachments.map((att) => (
                      <li key={att.id} className="flex items-center gap-2">
                        <span className="badge">{att.mimeType}</span>
                        <a className="text-emerald-300 hover:underline" href={att.url} target="_blank" rel="noreferrer">
                          {att.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {faq.shares.length > 0 && (
                <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 space-y-1">
                  <p className="text-xs font-semibold text-slate-300">Acessos compartilhados</p>
                  <ul className="text-sm space-y-1">
                    {faq.shares.map((share) => (
                      <li key={share.id} className="flex items-center gap-2">
                        <span className="badge">{share.permission}</span>
                        <span>{share.user.email}</span>
                        {share.expiresAt && (
                          <span className="text-xs text-slate-500">
                            expira em {new Date(share.expiresAt).toLocaleDateString('pt-BR')}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
          ))}
          {faqs.length === 0 && <p className="text-slate-400">Nenhuma FAQ encontrada com os filtros atuais.</p>}
        </div>
      </section>
    </main>
  );
}
