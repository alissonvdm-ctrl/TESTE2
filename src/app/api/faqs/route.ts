import { NextRequest, NextResponse } from 'next/server';
import { Permission, PublishStatus } from '@prisma/client';
import { prisma } from '@/lib/prisma';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

type IncomingAttachment = {
  name: string;
  url: string;
  mimeType?: string;
  size?: number;
};

type IncomingShare = {
  email: string;
  permission?: Permission;
  expiresAt?: string;
};

function normalizeStatus(value?: string | null): PublishStatus {
  return value === 'PUBLISHED' ? 'PUBLISHED' : 'DRAFT';
}

function normalizePermission(value?: string | null): Permission {
  if (value === 'EDIT' || value === 'ADMIN') return value;
  return 'VIEW';
}

async function ensureUser(email: string, name?: string) {
  return prisma.user.upsert({
    where: { email },
    update: { name: name ?? email },
    create: {
      email,
      name: name ?? email,
      role: 'EDITOR'
    }
  });
}

function parseTopics(param?: string | string[] | null) {
  if (!param) return [];
  const raw = Array.isArray(param) ? param : param.split(',');
  return raw.map((topic) => topic.trim()).filter(Boolean);
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get('q');
  const topicNames = parseTopics(searchParams.get('topics'));

  const faqs = await prisma.fAQ.findMany({
    where: {
      ...(q
        ? {
            OR: [
              { title: { contains: q, mode: 'insensitive' } },
              { summary: { contains: q, mode: 'insensitive' } },
              { content: { contains: q, mode: 'insensitive' } }
            ]
          }
        : {}),
      ...(topicNames.length
        ? {
            topics: {
              some: { topic: { name: { in: topicNames, mode: 'insensitive' } } }
            }
          }
        : {})
    },
    include: {
      topics: { include: { topic: true } },
      attachments: true,
      shares: { include: { user: true } },
      parent: { select: { id: true, title: true } },
      _count: { select: { children: true } }
    },
    orderBy: { updatedAt: 'desc' }
  });

  return NextResponse.json(
    faqs.map((faq) => ({
      ...faq,
      topics: faq.topics.map((t) => t.topic)
    }))
  );
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const {
    title,
    summary,
    content,
    authorEmail,
    authorName,
    topicNames = [],
    status = 'DRAFT',
    attachments = [],
    parentId,
    shares = []
  } = body as {
    title?: string;
    summary?: string | null;
    content?: string;
    authorEmail?: string;
    authorName?: string;
    topicNames?: string[];
    status?: string;
    attachments?: IncomingAttachment[];
    parentId?: string | null;
    shares?: IncomingShare[];
  };

  if (!title || !content) {
    return NextResponse.json({ message: 'title and content are required' }, { status: 400 });
  }

  if (parentId) {
    const parentExists = await prisma.fAQ.findUnique({ where: { id: parentId } });
    if (!parentExists) {
      return NextResponse.json({ message: 'Parent FAQ not found' }, { status: 400 });
    }
  }

  const author = await ensureUser(authorEmail ?? 'autor.demo@faq.local', authorName ?? 'Autor Demo');

  const topics = await Promise.all(
    topicNames.map((name: string) =>
      prisma.topic.upsert({
        where: { name },
        update: {},
        create: { name }
      })
    )
  );

  const attachmentsData = attachments
    .filter((att) => att.url && att.name)
    .map((att) => ({
      name: att.name,
      url: att.url,
      mimeType: att.mimeType ?? 'application/octet-stream',
      size: att.size ? Number(att.size) : 0
    }));

  const validShares = shares.filter((share) => share.email);

  const shareUsers = await Promise.all(
    validShares.map((share) => ensureUser(share.email, share.email.split('@')[0]))
  );

  const sharePayload = validShares.map((share, idx) => {
    const expiresAt = share.expiresAt;
    return {
      userId: shareUsers[idx].id,
      permission: normalizePermission(share.permission),
      expiresAt: expiresAt ? new Date(expiresAt) : null
    };
  });

  const faq = await prisma.fAQ.create({
    data: {
      title,
      summary,
      content,
      status: normalizeStatus(status),
      parentId: parentId ?? null,
      authorId: author.id,
      topics: {
        create: topics.map((topic) => ({ topicId: topic.id }))
      },
      attachments: {
        create: attachmentsData
      },
      shares: {
        create: sharePayload
      }
    },
    include: {
      topics: { include: { topic: true } },
      attachments: true,
      shares: { include: { user: true } },
      parent: { select: { id: true, title: true } },
      _count: { select: { children: true } }
    }
  });

  return NextResponse.json(
    { ...faq, topics: faq.topics.map((t) => t.topic) },
    { status: 201 }
  );
}
