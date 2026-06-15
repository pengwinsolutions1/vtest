// Saves an uploaded File to ./data/uploads and returns a public URL Replicate
// can fetch. Replicate's workers run in their own cloud; they need a URL
// reachable from the public internet — not a localhost path.
//
// Locally that means PUBLIC_URL must be a tunnel (ngrok, cloudflared, etc).
// In production it's just the deployed domain.
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ulid } from 'ulid';

const DATA_DIR = process.env.DATA_DIR || './data';
const UPLOAD_DIR = join(DATA_DIR, 'uploads');

const ALLOWED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const MAX_BYTES = 12 * 1024 * 1024; // 12 MB — phone photos are big

export async function saveUpload(file: File, kind: 'selfie' | 'garment'): Promise<string> {
  if (!ALLOWED_TYPES.has(file.type)) {
    throw new Error(`unsupported file type ${file.type} (need PNG, JPEG, or WebP)`);
  }
  if (file.size > MAX_BYTES) {
    throw new Error(`file too large (${(file.size / 1e6).toFixed(1)} MB > 12 MB)`);
  }
  const ext = file.type === 'image/png' ? 'png' : file.type === 'image/webp' ? 'webp' : 'jpg';
  const name = `${kind}_${ulid()}.${ext}`;
  const buf = Buffer.from(await file.arrayBuffer());
  await writeFile(join(UPLOAD_DIR, name), buf);
  // Public URL — served by app/uploads/[name]/route.ts
  const base = (process.env.PUBLIC_URL || 'http://localhost:3000').replace(/\/$/, '');
  return `${base}/uploads/${name}`;
}
