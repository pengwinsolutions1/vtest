// GET /api/jobs/[id] — polled by the frontend to track progress.
// On every poll, also nudges the orchestrator to check Replicate for completed
// predictions and advance the job state machine.
import { NextRequest, NextResponse } from 'next/server';
import { getJob } from '@/lib/db';
import { advanceJob } from '@/lib/orchestrator';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  // Best-effort advance; failure shouldn't block returning current state.
  await advanceJob(id).catch(() => {});
  const job = getJob(id);
  if (!job) return NextResponse.json({ error: 'not found' }, { status: 404 });
  return NextResponse.json({
    id: job.id,
    status: job.status,
    still_image_url: job.still_image_url,
    video_url: job.video_url,
    error: job.error,
    created: job.created,
    updated: job.updated,
  });
}
