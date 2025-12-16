import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET(_: Request, { params }: { params: { id: string } }) {
  const faq = await prisma.fAQ.findUnique({
    where: { id: params.id },
    include: { topics: { include: { topic: true } }, attachments: true }
  });

  if (!faq) {
    return NextResponse.json({ message: 'FAQ not found' }, { status: 404 });
  }

  return NextResponse.json({ ...faq, topics: faq.topics.map((t) => t.topic) });
}

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const body = await request.json();
  const { title, summary, content, status, topicNames = [] } = body;

  const topics = await Promise.all(
    topicNames.map((name: string) =>
      prisma.topic.upsert({
        where: { name },
        update: {},
        create: { name }
      })
    )
  );

  const faq = await prisma.fAQ.update({
    where: { id: params.id },
    data: {
      title,
      summary,
      content,
      status,
      topics: {
        deleteMany: {},
        create: topics.map((topic) => ({ topicId: topic.id }))
      }
    },
    include: { topics: { include: { topic: true } } }
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
