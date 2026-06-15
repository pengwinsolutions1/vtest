// POST /api/admin/garments — admin uploads a new garment from front + back PNGs.
// Saves both images, runs build_2view_glb.py to produce a GLB, inserts a catalog row.
//
// Multipart body:
//   front:    File   (PNG/JPEG/WEBP, the front view of the garment)
//   back:     File   (PNG/JPEG/WEBP, the back view of the garment)
//   name:     string (display name)
//   gender:   "men" | "women" | "unisex"
//   category: "top" | "bottom" | "dress"
//
// Returns: { id, name, glb_url }
//
// SECURITY NOTE: This endpoint has no auth. In production wire it behind a
// vendor-admin login (the old kiosk admin pattern, or whatever auth Next session
// you adopt). For dev / single-vendor box deployment, LAN-scope is fine.
import { NextRequest, NextResponse } from 'next/server';
import { ulid } from 'ulid';
import { writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { join, resolve } from 'node:path';
import { mkdirSync, existsSync } from 'node:fs';
import { db } from '@/lib/db';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const exec = promisify(execFile);
const DATA_DIR = process.env.DATA_DIR || './data';
const GARMENTS_DIR = join(DATA_DIR, 'uploads', 'garments');

// Production builder: Hunyuan3D-2mini shape + cylindrical UV with the real
// front+back photos as texture. ~8 min per garment on M1 Pro 16 GB.
// Lives in the sibling venv-hunyuan venv (separate from venv to avoid
// transformers version conflict with TripoSR).
const BUILDER_PY = resolve(process.cwd(), '..', 'ai-local', 'build_hunyuan_glb.py');
const BUILDER_PYTHON = resolve(process.cwd(), '..', 'ai-local', 'venv-hunyuan', 'bin', 'python');
// Long timeout for Hunyuan diffusion. 20 min covers worst-case M1 Pro runs.
const BUILD_TIMEOUT_MS = 20 * 60_000;

const ALLOWED_GENDER = new Set(['men', 'women', 'unisex']);
const ALLOWED_CATEGORY = new Set(['top', 'bottom', 'dress']);
const ALLOWED_MIME = new Set(['image/png', 'image/jpeg', 'image/webp']);
const MAX_BYTES = 12 * 1024 * 1024;

export async function POST(req: NextRequest) {
  let form: FormData;
  try { form = await req.formData(); }
  catch { return NextResponse.json({ error: 'expected multipart/form-data' }, { status: 400 }); }

  const front = form.get('front');
  const back = form.get('back');
  const name = String(form.get('name') ?? '').trim();
  const gender = String(form.get('gender') ?? '');
  const category = String(form.get('category') ?? '');

  if (!(front instanceof File) || !(back instanceof File)) {
    return NextResponse.json({ error: 'front and back files are required' }, { status: 400 });
  }
  if (!name) return NextResponse.json({ error: 'name is required' }, { status: 400 });
  if (!ALLOWED_GENDER.has(gender)) {
    return NextResponse.json({ error: `gender must be one of: ${[...ALLOWED_GENDER].join(', ')}` }, { status: 400 });
  }
  if (!ALLOWED_CATEGORY.has(category)) {
    return NextResponse.json({ error: `category must be one of: ${[...ALLOWED_CATEGORY].join(', ')}` }, { status: 400 });
  }
  for (const [label, f] of [['front', front], ['back', back]] as const) {
    if (!ALLOWED_MIME.has(f.type)) {
      return NextResponse.json({ error: `${label} has unsupported type ${f.type}` }, { status: 400 });
    }
    if (f.size > MAX_BYTES) {
      return NextResponse.json({ error: `${label} too large (${(f.size / 1e6).toFixed(1)} MB > 12 MB)` }, { status: 400 });
    }
  }

  // Generate a stable id from name + gender + ULID suffix to keep filenames sortable + unique.
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const id = `${gender}s-${slug}-${ulid().slice(-6).toLowerCase()}`;

  mkdirSync(GARMENTS_DIR, { recursive: true });

  // Save the two source images
  const ext = (f: File) => f.type === 'image/png' ? 'png' : f.type === 'image/webp' ? 'webp' : 'jpg';
  const frontName = `${id}-front.${ext(front)}`;
  const backName  = `${id}-back.${ext(back)}`;
  const glbName   = `${id}.glb`;
  const frontPath = join(GARMENTS_DIR, frontName);
  const backPath  = join(GARMENTS_DIR, backName);
  const glbPath   = join(GARMENTS_DIR, glbName);

  try {
    await writeFile(frontPath, Buffer.from(await front.arrayBuffer()));
    await writeFile(backPath, Buffer.from(await back.arrayBuffer()));
  } catch (e: any) {
    return NextResponse.json({ error: `failed to save uploads: ${e.message}` }, { status: 500 });
  }

  // Build the GLB. ~5 seconds. Synchronous so we can return the final url.
  // If the builder is missing (ai-local not installed yet), bail with a clear error.
  if (!existsSync(BUILDER_PYTHON) || !existsSync(BUILDER_PY)) {
    return NextResponse.json({
      error: `Hunyuan builder not found. Expected: ${BUILDER_PY} (using ${BUILDER_PYTHON}). See ai-local/README.md.`,
    }, { status: 500 });
  }
  try {
    await exec(BUILDER_PYTHON, [BUILDER_PY, frontPath, backPath, glbPath, '--steps', '25'], {
      maxBuffer: 32 * 1024 * 1024,
      timeout: BUILD_TIMEOUT_MS,
    });
  } catch (e: any) {
    return NextResponse.json({ error: `GLB build failed: ${e.stderr || e.message}` }, { status: 500 });
  }

  // Insert catalog row. Use the front image as the thumbnail.
  const now = Date.now();
  db.prepare(`INSERT INTO garments
    (id, name, filename, category, gender, glb_filename, glb_status, created)
    VALUES (?, ?, ?, ?, ?, ?, 'ready', ?)`
  ).run(id, name, frontName, category, gender, glbName, now);

  const base = (process.env.PUBLIC_URL || 'http://localhost:3000').replace(/\/$/, '');
  return NextResponse.json({
    id,
    name,
    category,
    gender,
    url: `${base}/uploads/garments/${frontName}`,
    glb_url: `${base}/uploads/garments/${glbName}`,
  });
}
