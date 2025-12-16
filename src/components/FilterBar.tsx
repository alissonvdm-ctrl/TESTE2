"use client";

import { FormEvent, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

type FilterBarProps = {
  topics: { id: string; name: string; _count: { faqs: number } }[];
  initialSearch?: string;
  selectedTopics?: string[];
};

export default function FilterBar({ topics, initialSearch = '', selectedTopics = [] }: FilterBarProps) {
  const router = useRouter();
  const params = useSearchParams();
  const [query, setQuery] = useState(initialSearch);
  const [activeTopics, setActiveTopics] = useState<string[]>(selectedTopics);

  useEffect(() => setQuery(initialSearch), [initialSearch]);
  useEffect(() => setActiveTopics(selectedTopics), [selectedTopics.join(',')]);

  const topicNames = useMemo(
    () => new Set(activeTopics.map((topic) => topic.toLowerCase())),
    [activeTopics]
  );

  function toggleTopic(name: string) {
    const normalized = name.toLowerCase();
    const next = new Set(topicNames);
    if (next.has(normalized)) {
      next.delete(normalized);
    } else {
      next.add(normalized);
    }
    setActiveTopics(Array.from(next));
  }

  function applyFilters(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const searchParams = new URLSearchParams(params.toString());

    if (query.trim()) searchParams.set('q', query.trim());
    else searchParams.delete('q');

    if (topicNames.size > 0) searchParams.set('topics', Array.from(topicNames).join(','));
    else searchParams.delete('topics');

    const path = searchParams.toString();
    router.push(path ? `/?${searchParams.toString()}` : '/');
  }

  function clearFilters() {
    setQuery('');
    setActiveTopics([]);
    router.push('/');
  }

  return (
    <form className="space-y-3" onSubmit={applyFilters}>
      <div className="grid gap-3 md:grid-cols-[1fr_220px]">
        <label className="flex flex-col gap-1 text-sm text-slate-200">
          Busca livre
          <input
            className="rounded-md border border-slate-800 bg-slate-900 px-3 py-2 text-slate-100"
            placeholder="Título, resumo ou conteúdo"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <div className="flex items-end gap-2">
          <button
            type="submit"
            className="rounded-lg bg-emerald-500 px-3 py-2 text-slate-900 font-semibold hover:bg-emerald-400"
          >
            Aplicar filtros
          </button>
          <button
            type="button"
            className="rounded-lg border border-slate-800 px-3 py-2 text-slate-100 hover:border-slate-600"
            onClick={clearFilters}
          >
            Limpar
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-sm">
        {topics.map((topic) => {
          const isActive = topicNames.has(topic.name.toLowerCase());
          return (
            <button
              key={topic.id}
              type="button"
              onClick={() => toggleTopic(topic.name)}
              className={`rounded-full border px-3 py-1 transition ${
                isActive
                  ? 'border-emerald-400 bg-emerald-500/10 text-emerald-200'
                  : 'border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-500'
              }`}
            >
              {topic.name} <span className="text-xs text-slate-400">({topic._count.faqs})</span>
            </button>
          );
        })}
        {topics.length === 0 && <span className="text-slate-400">Nenhum assunto cadastrado.</span>}
      </div>
    </form>
  );
}
