# Sistema Web de FAQ

Aplicação Next.js pronta para deploy na Vercel com banco de dados PostgreSQL (Neon/Supabase/RDS) usando Prisma. Inclui CRUD de FAQs, categorização por assuntos, hierarquia, anexos e compartilhamento.

## Estrutura de Pastas
- `src/app` — rotas e páginas (App Router) e componentes de interface.
- `src/app/api` — handlers serverless para CRUD de FAQs.
- `src/components/FAQForm.tsx` — formulário client-side para criar FAQs.
- `src/lib/prisma.ts` — singleton do cliente Prisma para evitar conexões extras.
- `prisma/schema.prisma` — modelagem completa do banco.

## Como rodar localmente
1. Instale dependências: `npm install`.
2. Crie o banco e sincronize o schema: `npx prisma db push`.
3. Rode o servidor: `npm run dev` e acesse `http://localhost:3000`.

## Deploy na Vercel
1. Configure variáveis de ambiente no dashboard da Vercel:
   - `DATABASE_URL` — string PostgreSQL (preferencialmente Neon para conexão sem TLS local).
   - `NEXTAUTH_SECRET` — qualquer string segura para criptografia de cookies (mesmo sem auth configurada ainda).
2. Ative o "Prisma Accelerate" ou "Prisma Data Proxy" se desejar pool para conexões serverless.
3. Faça deploy com `vercel` CLI ou conectando o repositório ao GitHub.

## Rotas de API
- `GET /api/faqs` — lista FAQs com assuntos e anexos.
- `POST /api/faqs` — cria FAQ com `title`, `summary`, `content`, `status`, `topicNames[]`, `authorId`.
- `GET /api/faqs/:id` — detalhes da FAQ.
- `PATCH /api/faqs/:id` — atualiza campos e assuntos.
- `DELETE /api/faqs/:id` — remove FAQ e registros relacionados.

## Modelagem do Banco (Prisma)
- **User**: nome, email único, role (`ADMIN`, `EDITOR`, `READER`).
- **FAQ**: título, resumo, conteúdo, status de publicação, prioridade, relação pai/filho.
- **Topic** e **FAQTopic**: assuntos dinâmicos por FAQ.
- **Attachment**: metadados de arquivos (URL externa ou bucket da Vercel Storage/S3).
- **Share**: permissões específicas por FAQ (`VIEW`, `EDIT`, `ADMIN`) com expiração opcional.

## Segurança e anexos
- Validação de MIME/size deve ser feita antes de gravar a URL em `Attachment`.
- Utilize storage externo (S3, R2 ou Vercel Blob) com URLs assinadas.
- Log de auditoria pode ser adicionado via middleware de API ou Prisma middleware.

## Roadmap sugerido
- Autenticação (NextAuth) para vincular `authorId` ao usuário logado.
- Full-text search (Postgres `tsvector` ou serviço como Meilisearch/Algolia).
- Interface para upload de anexos com antivírus e compressão de imagem.
- Compartilhamento em massa por times e cache de ACL em Redis.
