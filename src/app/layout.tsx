import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FAQ Workspace',
  description: 'Sistema de FAQ pronto para Vercel com Prisma'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="bg-slate-950 text-slate-100">
        <div className="min-h-screen mx-auto max-w-5xl px-6 py-10 font-sans">
          <header className="flex items-center justify-between mb-10">
            <div>
              <p className="text-sm text-slate-400">FAQ Central</p>
              <h1 className="text-2xl font-semibold">Base de conhecimento</h1>
            </div>
            <a
              className="text-sm text-emerald-300 hover:text-emerald-200"
              href="https://vercel.com/docs"
              target="_blank"
              rel="noreferrer"
            >
              Documentação Vercel
            </a>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
