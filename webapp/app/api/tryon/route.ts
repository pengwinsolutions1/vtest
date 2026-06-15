// POST /api/tryon — start a try-on job from a garment_id (catalog pick) +
// selfie file.
//
// The garment's `category` (top/bottom/dress) is fetched from the catalog and
// routes the AI backend to the correct model variant. This is what gives us
// "auto-detect": admin picked the category at upload time, every try-on uses it.
import { NextRequest, NextResponse } from 'next/server';
import { ulid } from 'ulid';
import { saveUpload } from '@/lib/uploads';
import { createJob, getJob, jobsByIpInLast, getGarment } from '@/lib/db';
import { kickoffJob } from '@/lib/orchestrator';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const PER_IP_LIMIT = 10;
const PER_IP_WINDOW = 60 * 60_000;

export async function POST(req: NextRequest) {
  const ip = (req.headers.get('x-forwarded-for') || '').split(',')[0].trim() || 'unknown';
  if (jobsByIpInLast(ip, PER_IP_WINDOW) >= PER_IP_LIMIT) {
    return NextResponse.json({ error: 'rate limit: max 10 try-ons per hour' }, { status: 429 });
  }

  let form: FormData;
  try { form = await req.formData(); }
  catch { return NextResponse.json({ error: 'expected multipart/form-data' }, { status: 400 }); }

  const selfie = form.get('selfie');
  if (!(selfie instanceof File)) {
    return NextResponse.json({ error: 'selfie file is required' }, { status: 400 });
  }

  const garmentId = form.get('garment_id');
  if (typeof garmentId !== 'string' || !garmentId) {
    return NextResponse.json({ error: 'garment_id is required (pick one from /api/garments)' }, { status: 400 });
  }
  const garment = getGarment(garmentId);
  if (!garment) {
    return NextResponse.json({ error: `unknown garment_id: ${garmentId}` }, { status: 400 });
  }

  const base = (process.env.PUBLIC_URL || 'http://localhost:3000').replace(/\/$/, '');
  const garment_url = `${base}/uploads/garments/${garment.filename}`;

  let selfie_url: string;
  try { selfie_url = await saveUpload(selfie, 'selfie'); }
  catch (e: any) { return NextResponse.json({ error: e.message }, { status: 400 }); }

  const id = ulid();
  createJob({
    id, selfie_url, garment_url,
    client_ip: ip,
    user_agent: req.headers.get('user-agent') ?? undefined,
  });

  try {
    await kickoffJob(getJob(id)!, garment.category);
  } catch (e: any) {
    return NextResponse.json({ error: `AI backend kickoff failed: ${e.message}`, id }, { status: 502 });
  }

  return NextResponse.json({ id, garment: { name: garment.name, category: garment.category } });
}
