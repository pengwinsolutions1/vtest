// Serves catalog garment images at /uploads/garments/<name>.
// Same shape as /uploads/[name]/route.ts but reads from the garments subdir.
import { NextRequest } from 'next/server';
import { createReadStream, statSync } from 'node:fs';
import { join, basename } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const GARMENT_DIR = join(DATA_DIR, 'uploads', 'garments');

export async function GET(_req: NextRequest, ctx: { params: Promise<{ name: string }> }) {
  const { name } = await ctx.params;
  const safe = basename(name);
  const path = join(GARMENT_DIR, safe);
  try {
    const st = statSync(path);
    const stream = createReadStream(path);
    const ext = safe.split('.').pop()?.toLowerCase();
    const type =
      ext === 'png'  ? 'image/png'  :
      ext === 'webp' ? 'image/webp' :
      ext === 'glb'  ? 'model/gltf-binary' :
      ext === 'gltf' ? 'model/gltf+json'   :
                       'image/jpeg';
    return new Response(stream as any, {
      headers: {
        'Content-Type': type,
        'Content-Length': String(st.size),
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch {
    return new Response('not found', { status: 404 });
  }
}
