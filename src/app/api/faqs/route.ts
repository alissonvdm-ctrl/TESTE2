import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET() {
  const faqs = await prisma.fAQ.findMany({
    include: { topics: { include: { topic: true } }, attachments: true },
    orderBy: { updatedAt: 'desc' }
  });

  return NextResponse.json(
    faqs.map((faq) => ({
      ...faq,
      topics: faq.topics.map((t) => t.topic)
    }))
  );
}

export async function POST(request: Request) {
  const body = await request.json();
  const { title, summary, content, authorId, topicNames = [], status = 'DRAFT' } = body;

  const topics = await Promise.all(
    topicNames.map((name: string) =>
      prisma.topic.upsert({
        where: { name },
        update: {},
        create: { name }
      })
    )
  );

  const faq = await prisma.fAQ.create({
    data: {
      title,
      summary,
      content,
      status,
      authorId,
      topics: {
        create: topics.map((topic) => ({ topicId: topic.id }))
      }
    },
    include: { topics: { include: { topic: true } } }
  });

  return NextResponse.json(
    { ...faq, topics: faq.topics.map((t) => t.topic) },
    { status: 201 }
  );
}
