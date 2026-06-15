'use client';
// Admin upload page — drop a front PNG + back PNG, get a 3D garment in the catalog.
// Designed for a single vendor admin on the same LAN as the box. No auth yet.
//
// POSTs to /api/admin/garments which runs the 2-view → GLB builder synchronously
// (~5s) and inserts a catalog row.
import { useState, useRef } from 'react';

type Gender = 'men' | 'women' | 'unisex';
type Category = 'top' | 'bottom' | 'dress';

interface Slot { file: File | null; preview: string | null; }
const empty = (): Slot => ({ file: null, preview: null });

export default function AdminUpload() {
  const [front, setFront] = useState<Slot>(empty());
  const [back,  setBack]  = useState<Slot>(empty());
  const [name, setName] = useState('');
  const [gender, setGender] = useState<Gender>('men');
  const [category, setCategory] = useState<Category>('top');
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const frontRef = useRef<HTMLInputElement>(null);
  const backRef  = useRef<HTMLInputElement>(null);

  function pick(setter: (s: Slot) => void, f: File) {
    setter({ file: f, preview: URL.createObjectURL(f) });
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!front.file || !back.file || !name.trim()) return;
    setSubmitting(true); setError(null);
    setStatus('Uploading photos…');

    // Hunyuan3D-2 takes ~8 minutes on M1 Pro. Drive a soft progress message
    // while we wait so the admin knows the box isn't hung. Updated every 20s.
    const startedAt = Date.now();
    const progressTick = setInterval(() => {
      const min = Math.floor((Date.now() - startedAt) / 60_000);
      const expected = 9; // rough total minutes including upload + decimation
      setStatus(`Building 3D mesh… (~${min}/${expected} min — Hunyuan diffusion is running)`);
    }, 20_000);

    try {
      const fd = new FormData();
      fd.append('front', front.file);
      fd.append('back', back.file);
      fd.append('name', name.trim());
      fd.append('gender', gender);
      fd.append('category', category);
      setStatus('Building 3D mesh… (this takes ~8 minutes)');
      const r = await fetch('/api/admin/garments', { method: 'POST', body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || `HTTP ${r.status}`);
      setStatus(`Added "${data.name}" to the catalog. Refresh the main app to see it.`);
      setFront(empty()); setBack(empty()); setName('');
    } catch (e: any) {
      setError(e.message);
      setStatus(null);
    } finally {
      clearInterval(progressTick);
      setSubmitting(false);
    }
  }

  const ready = !!(front.file && back.file && name.trim()) && !submitting;

  return (
    <div className="admin">
      <h1>Add a garment</h1>
      <p className="sub">Upload front + back photos. We&apos;ll build the 3D model and add it to the live AR catalog.</p>

      {error && <div className="err">{error}</div>}
      {status && !error && <div className="status">{status}</div>}

      <form onSubmit={submit}>
        <div className="row">
          <Slot
            label="Front view"
            slot={front}
            inputRef={frontRef}
            onPick={(f) => pick(setFront, f)}
            onClear={() => setFront(empty())}
          />
          <Slot
            label="Back view"
            slot={back}
            inputRef={backRef}
            onPick={(f) => pick(setBack, f)}
            onClear={() => setBack(empty())}
          />
        </div>

        <label>Name
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Black Cocktail Dress" />
        </label>

        <div className="row">
          <label>Gender
            <select value={gender} onChange={(e) => setGender(e.target.value as Gender)}>
              <option value="men">Men</option>
              <option value="women">Women</option>
              <option value="unisex">Unisex</option>
            </select>
          </label>
          <label>Category
            <select value={category} onChange={(e) => setCategory(e.target.value as Category)}>
              <option value="top">Top</option>
              <option value="bottom">Bottom</option>
              <option value="dress">Dress</option>
            </select>
          </label>
        </div>

        <button className="cta" type="submit" disabled={!ready}>
          {submitting ? 'Working…' : 'Add to catalog →'}
        </button>
      </form>

      <p className="foot">
        Front + back images should be product shots on a plain background. Background gets removed automatically.
        Build runs locally on the vendor box; nothing leaves the LAN.
      </p>

      <style jsx>{`
        .admin { max-width: 720px; margin: 0 auto; padding: 32px 20px 64px; min-height: 100dvh; }
        h1 { margin: 0 0 4px; font-size: 28px; font-weight: 700; }
        .sub { margin: 0 0 24px; color: var(--muted); font-size: 15px; }
        .err { background: #1f0d0d; border: 1px solid var(--error); color: #fca5a5; padding: 12px 14px; border-radius: 10px; margin-bottom: 16px; }
        .status { background: rgba(99,102,241,0.12); border: 1px solid var(--accent); color: #c7d2fe; padding: 12px 14px; border-radius: 10px; margin-bottom: 16px; }
        form { display: flex; flex-direction: column; gap: 16px; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
        input, select { padding: 12px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; color: var(--fg); font-size: 16px; text-transform: none; }
        .foot { margin-top: 24px; color: var(--muted); font-size: 12px; line-height: 1.6; }
      `}</style>
    </div>
  );
}

function Slot({
  label, slot, inputRef, onPick, onClear,
}: {
  label: string;
  slot: Slot;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onPick: (f: File) => void;
  onClear: () => void;
}) {
  return (
    <div
      className="upload-slot"
      onClick={() => !slot.preview && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); }}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f) onPick(f);
      }}
    >
      <div className="slot-label">{label}</div>
      {slot.preview ? (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={slot.preview} alt={label} />
          <button type="button" className="clear" onClick={(e) => { e.stopPropagation(); onClear(); }}>Change</button>
        </>
      ) : (
        <div className="placeholder">
          <div className="icon">⬆️</div>
          <div>Click or drop image</div>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onPick(f); }}
        style={{ display: 'none' }}
      />
      <style jsx>{`
        .upload-slot {
          position: relative; cursor: pointer;
          background: var(--card); border: 1px dashed var(--border); border-radius: 14px;
          padding: 14px; min-height: 260px;
          display: flex; flex-direction: column; align-items: center; gap: 10px;
          transition: border-color 0.15s;
        }
        .upload-slot:hover { border-color: var(--accent); }
        .slot-label { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; align-self: flex-start; }
        .placeholder { color: var(--muted); font-size: 14px; text-align: center; padding: 40px 0; }
        .placeholder .icon { font-size: 36px; margin-bottom: 8px; }
        img { width: 100%; max-height: 260px; object-fit: contain; border-radius: 8px; background: #fff; }
        .clear { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: #fff; border: 0; border-radius: 999px; padding: 6px 12px; font-size: 12px; cursor: pointer; }
      `}</style>
    </div>
  );
}
