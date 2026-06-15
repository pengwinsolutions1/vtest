// Serves uploaded selfie/garment files. We don't trust Next's static handling
// for user uploads (it stages everything at build time); a streaming route
// lets us serve files written at runtime without rebuilding.
import { NextRequest } from 'next/server';
import { createReadStream, statSync } from 'node:fs';
import { join, basename } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const UPLOAD_DIR = join(DATA_DIR, 'uploads');

export async function GET(_req: NextRequest, ctx: { params: Promise<{ name: string }> }) {
  const { name } = await ctx.params;
  // basename() strips any traversal attempt (../../etc/passwd)
  const safe = basename(name);
  const path = join(UPLOAD_DIR, safe);
  try {
    const st = statSync(path);
    const stream = createReadStream(path);
    const ext = safe.split('.').pop()?.toLowerCase();
    const type = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
    return new Response(stream as any, {
      headers: {
        'Content-Type': type,
        'Content-Length': String(st.size),
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch {
    return new Response('not found', { status: 404 });
  }
}
