import { NextResponse } from 'next/server';
import { Permission, PublishStatus } from '@prisma/client';
import { prisma } from '@/lib/prisma';

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

export async function GET(_: Request, { params }: { params: { id: string } }) {
  const faq = await prisma.fAQ.findUnique({
    where: { id: params.id },
    include: {
      topics: { include: { topic: true } },
      attachments: true,
      shares: { include: { user: true } },
      parent: { select: { id: true, title: true } },
      _count: { select: { children: true } }
    }
  });

  if (!faq) {
    return NextResponse.json({ message: 'FAQ not found' }, { status: 404 });
  }

  return NextResponse.json({ ...faq, topics: faq.topics.map((t) => t.topic) });
}

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const body = await request.json();
  const {
    title,
    summary,
    content,
    status,
    topicNames = [],
    attachments = [],
    parentId,
    shares = []
  } = body as {
    title?: string;
    summary?: string | null;
    content?: string;
    status?: string;
    topicNames?: string[];
    attachments?: IncomingAttachment[];
    parentId?: string | null;
    shares?: IncomingShare[];
  };

  if (parentId) {
    if (parentId === params.id) {
      return NextResponse.json({ message: 'Parent cannot be the same FAQ' }, { status: 400 });
    }

    const parentExists = await prisma.fAQ.findUnique({ where: { id: parentId } });
    if (!parentExists) {
      return NextResponse.json({ message: 'Parent FAQ not found' }, { status: 400 });
    }
  }

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

  const faq = await prisma.$transaction(async (tx) => {
    await tx.attachment.deleteMany({ where: { faqId: params.id } });
    await tx.share.deleteMany({ where: { faqId: params.id } });

    return tx.fAQ.update({
      where: { id: params.id },
      data: {
        title,
        summary,
        content,
        status: normalizeStatus(status),
        parentId: parentId ?? null,
        topics: {
          deleteMany: {},
          create: topics.map((topic) => ({ topicId: topic.id }))
        },
        attachments: {
          create: attachmentsData
        },
        shares: {
          create: shareUsers.map((user, idx) => ({
            userId: user.id,
            permission: normalizePermission(validShares[idx].permission),
            expiresAt: validShares[idx].expiresAt ? new Date(validShares[idx].expiresAt) : null
          }))
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
  });

  return NextResponse.json({ ...faq, topics: faq.topics.map((t) => t.topic) });
}

export async function DELETE(_: Request, { params }: { params: { id: string } }) {
  await prisma.attachment.deleteMany({ where: { faqId: params.id } });
  await prisma.share.deleteMany({ where: { faqId: params.id } });
  await prisma.fAQTopic.deleteMany({ where: { faqId: params.id } });
  await prisma.fAQ.delete({ where: { id: params.id } });
  return NextResponse.json({ success: true });
}
