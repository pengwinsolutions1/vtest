// SQLite store for try-on jobs + garment catalog. Single-process, single-file,
// durable enough for production at the scales we care about.
// Swap to Postgres later by reimplementing the same surface.
import Database from 'better-sqlite3';
import { mkdirSync, copyFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
mkdirSync(DATA_DIR, { recursive: true });
mkdirSync(join(DATA_DIR, 'uploads'), { recursive: true });
mkdirSync(join(DATA_DIR, 'uploads', 'garments'), { recursive: true });

const db = new Database(join(DATA_DIR, 'tryon.db'));
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL,          -- queued | still | video | succeeded | failed
    selfie_url      TEXT NOT NULL,          -- public URL of user's selfie
    garment_url     TEXT NOT NULL,          -- public URL of garment image
    still_prediction_id  TEXT,              -- backend prediction id for try-on still
    still_image_url      TEXT,              -- result of step 1
    video_prediction_id  TEXT,              -- backend prediction id for image-to-video
    video_url            TEXT,              -- final MP4 url
    error           TEXT,                   -- failure reason
    created         INTEGER NOT NULL,
    updated         INTEGER NOT NULL,
    client_ip       TEXT,                   -- crude rate limit / abuse signal
    user_agent      TEXT
  );
  CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created DESC);
  CREATE INDEX IF NOT EXISTS jobs_still   ON jobs(still_prediction_id);
  CREATE INDEX IF NOT EXISTS jobs_video   ON jobs(video_prediction_id);

  CREATE TABLE IF NOT EXISTS garments (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    filename      TEXT NOT NULL UNIQUE,    -- PNG under data/uploads/garments/
    category      TEXT NOT NULL DEFAULT 'top',     -- top | bottom | dress
    gender        TEXT NOT NULL DEFAULT 'unisex',  -- men | women | unisex
    glb_filename  TEXT,                    -- optional .glb under data/uploads/garments/; live AR uses it instead of the procedural mesh when set
    glb_status    TEXT,                    -- null | 'pending' | 'ready' | 'failed' — only relevant when a conversion job is in flight
    created       INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS garments_created ON garments(created);
  CREATE INDEX IF NOT EXISTS garments_category ON garments(category);
  CREATE INDEX IF NOT EXISTS garments_gender ON garments(gender);
`);

// In-place migrations — additive columns are safe to ALTER on existing DBs.
try { db.prepare('SELECT category FROM garments LIMIT 1').get(); }
catch { db.exec("ALTER TABLE garments ADD COLUMN category TEXT NOT NULL DEFAULT 'top'"); }
try { db.prepare('SELECT gender FROM garments LIMIT 1').get(); }
catch { db.exec("ALTER TABLE garments ADD COLUMN gender TEXT NOT NULL DEFAULT 'unisex'"); }
try { db.prepare('SELECT glb_filename FROM garments LIMIT 1').get(); }
catch { db.exec("ALTER TABLE garments ADD COLUMN glb_filename TEXT"); }
try { db.prepare('SELECT glb_status FROM garments LIMIT 1').get(); }
catch { db.exec("ALTER TABLE garments ADD COLUMN glb_status TEXT"); }
db.exec(`
  CREATE INDEX IF NOT EXISTS garments_category ON garments(category);
  CREATE INDEX IF NOT EXISTS garments_gender ON garments(gender);
`);

// --- Seed the catalog from PNGs in ../seed-garments/ on first boot. Lets the
// demo open with a populated grid even before an admin uploads anything.
//
// Filename conventions the seed understands (admin can override at upload):
//   mens-*    → gender=men
//   womens-*  → gender=women
//   *-dress*  or *-gown*  or *-frock*       → category=dress
//   *-pant*, *-skirt*, *-jean*, *-shorts*   → category=bottom
//   anything else                            → category=top
//
// Falls back to ../ (project root) if seed-garments/ is empty — legacy path
// for the original sample PNGs.
function guessCategoryFromName(name: string): 'top' | 'bottom' | 'dress' {
  const n = name.toLowerCase();
  if (/(pant|trouser|short|jean|skirt|legging)/.test(n)) return 'bottom';
  if (/(dress|gown|frock|jumpsuit)/.test(n)) return 'dress';
  return 'top';
}
function guessGenderFromName(name: string): 'men' | 'women' | 'unisex' {
  const n = name.toLowerCase();
  if (/(^|[^a-z])womens?[-_]/.test(n) || /women/.test(n)) return 'women';
  if (/(^|[^a-z])mens?[-_]/.test(n) || /\bmens?\b/.test(n)) return 'men';
  return 'unisex';
}
function prettifyName(filename: string): string {
  // strip extension + men/women prefix so the display name is just the garment
  let s = filename.replace(/\.[^.]+$/, '');
  s = s.replace(/^(mens?|womens?)[-_]/i, '');
  return s.replace(/[-_]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function seedGarments() {
  const existing = (db.prepare('SELECT COUNT(*) AS c FROM garments').get() as { c: number }).c;
  if (existing > 0) return;

  // Curated set first; falls back to scattered PNGs at repo root.
  const candidates = ['./seed-garments/', '../seed-garments/', '../'];
  let copied = 0;
  for (const dir of candidates) {
    if (!existsSync(dir)) continue;
    const files = readdirSync(dir);
    const images = files.filter(f => /\.(png|jpe?g|webp)$/i.test(f));
    const glbs   = files.filter(f => /\.glb$/i.test(f));

    // Index GLBs by their base name (strip extension) so we can pair them with
    // an image of the same base (e.g. mens-white-tshirt.png + mens-white-tshirt.glb).
    const glbByBase: Record<string, string> = {};
    for (const g of glbs) glbByBase[g.replace(/\.glb$/i, '').toLowerCase()] = g;

    for (const f of images) {
      const src = join(dir, f);
      const dst = join(DATA_DIR, 'uploads', 'garments', f);
      try {
        copyFileSync(src, dst);
        const base = f.replace(/\.[^.]+$/, '');
        const id = base.replace(/[^a-z0-9-]+/gi, '-').toLowerCase();
        const name = prettifyName(f);
        const category = guessCategoryFromName(f);
        const gender = guessGenderFromName(f);

        // Pair with a matching .glb if one sits next to it.
        const matchingGlb = glbByBase[base.toLowerCase()];
        let glbStatus: string | null = null;
        if (matchingGlb) {
          try {
            copyFileSync(join(dir, matchingGlb), join(DATA_DIR, 'uploads', 'garments', matchingGlb));
            glbStatus = 'ready';
          } catch (e: any) {
            console.warn(`[seed] failed to copy GLB ${matchingGlb}: ${e.message}`);
          }
        }

        db.prepare(`INSERT OR IGNORE INTO garments
          (id, name, filename, category, gender, glb_filename, glb_status, created)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
        ).run(id, name, f, category, gender, matchingGlb ?? null, glbStatus, Date.now());
        copied++;
      } catch (e: any) {
        console.warn(`[seed] skipped ${f}: ${e.message}`);
      }
    }
    if (copied > 0) break;
  }
  if (copied > 0) console.log(`[seed] copied ${copied} garment(s) into the catalog`);
}
seedGarments();

export type JobStatus = 'queued' | 'still' | 'video' | 'succeeded' | 'failed';

export interface Job {
  id: string;
  status: JobStatus;
  selfie_url: string;
  garment_url: string;
  still_prediction_id: string | null;
  still_image_url: string | null;
  video_prediction_id: string | null;
  video_url: string | null;
  error: string | null;
  created: number;
  updated: number;
  client_ip: string | null;
  user_agent: string | null;
}

export function createJob(input: {
  id: string;
  selfie_url: string;
  garment_url: string;
  client_ip?: string;
  user_agent?: string;
}): void {
  const now = Date.now();
  db.prepare(`
    INSERT INTO jobs (id, status, selfie_url, garment_url, created, updated, client_ip, user_agent)
    VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
  `).run(input.id, input.selfie_url, input.garment_url, now, now, input.client_ip ?? null, input.user_agent ?? null);
}

export function getJob(id: string): Job | null {
  return (db.prepare('SELECT * FROM jobs WHERE id = ?').get(id) as Job | undefined) ?? null;
}

export function getJobByPrediction(predictionId: string): Job | null {
  return (db.prepare(`
    SELECT * FROM jobs
    WHERE still_prediction_id = ? OR video_prediction_id = ?
    ORDER BY updated DESC LIMIT 1
  `).get(predictionId, predictionId) as Job | undefined) ?? null;
}

export function updateJob(id: string, patch: Partial<Job>): void {
  const fields = Object.keys(patch).filter(k => k !== 'id' && k !== 'created');
  if (!fields.length) return;
  const sql = `UPDATE jobs SET ${fields.map(f => `${f} = ?`).join(', ')}, updated = ? WHERE id = ?`;
  const values = fields.map(f => (patch as any)[f]);
  db.prepare(sql).run(...values, Date.now(), id);
}

// Rough per-IP rate limit so a single user can't burn $50 of Replicate credit
// while we're not looking. Read by /api/tryon before queueing a new job.
export function jobsByIpInLast(ip: string, windowMs: number): number {
  const since = Date.now() - windowMs;
  return (db.prepare('SELECT COUNT(*) AS c FROM jobs WHERE client_ip = ? AND created > ?')
    .get(ip, since) as { c: number }).c;
}

// ---------- Garment catalog ----------
export type GarmentCategory = 'top' | 'bottom' | 'dress';
export type GarmentGender = 'men' | 'women' | 'unisex';

export interface Garment {
  id: string;
  name: string;
  filename: string;
  category: GarmentCategory;
  gender: GarmentGender;
  glb_filename: string | null;
  glb_status: 'pending' | 'ready' | 'failed' | null;
  created: number;
}

export function listGarments(): Garment[] {
  return db.prepare('SELECT * FROM garments ORDER BY created').all() as Garment[];
}

export function getGarment(id: string): Garment | null {
  return (db.prepare('SELECT * FROM garments WHERE id = ?').get(id) as Garment | undefined) ?? null;
}

export { db };
