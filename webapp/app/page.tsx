'use client';
// Customer page — LIVE virtual try-on, nothing else.
//
// Webcam streams to the vendor-box's /ws/live WebSocket. The vendor box runs
// DM-VTON (~50ms/frame on RTX 4080) and returns each rendered frame. We
// display the rendered frame on a <canvas>. Customer sees themselves wearing
// the selected garment in real time as they move.
//
// No snapshots. No "Try On" button. No polling. No photo-booth flow.
// Picking a dress IS trying it on — instantly, live.
//
// The /api/tryon + /api/jobs endpoints are still in the codebase but unused
// by this page. They're kept for the admin upload tool and any future
// snapshot-mode product.

import { useEffect, useState } from 'react';
import LiveStream from './components/LiveStream';

type Category = 'top' | 'bottom' | 'dress';
type Gender = 'men' | 'women' | 'unisex';

interface Garment {
  id: string;
  name: string;
  url: string;
  glb_url?: string | null;
  category: Category;
  gender: Gender;
}

interface BackendInfo {
  live_available: boolean;
  live_mode: 'dm_vton' | 'idm_vton_lowstep' | 'unavailable';
  ws_url: string | null;
  backend: string;
  error?: string;
}

const GENDER_LABELS: Record<Gender, string> = { men: 'Men', women: 'Women', unisex: 'Unisex' };
const GENDER_EMOJI:  Record<Gender, string> = { men: '🧔', women: '👩', unisex: '🧑' };

export default function Home() {
  const [garments, setGarments] = useState<Garment[]>([]);
  const [activeGender, setActiveGender] = useState<Gender | 'all'>('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [backendInfo, setBackendInfo] = useState<BackendInfo | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);

  const selectedGarment = garments.find(g => g.id === selectedId) || null;

  // Fetch the garment catalog + AI backend capabilities on mount
  useEffect(() => {
    fetch('/api/garments')
      .then(r => r.json())
      .then(d => {
        const list: Garment[] = d.garments || [];
        setGarments(list);
        if (list.length) setSelectedId(list[0].id);
      })
      .catch(e => setFatal(`catalog load failed: ${e.message}`));

    fetch('/api/backend-info')
      .then(r => r.json())
      .then((info: BackendInfo) => setBackendInfo(info))
      .catch(e => setFatal(`backend-info failed: ${e.message}`));
  }, []);

  // ── Render branches ────────────────────────────────────────────────────
  if (fatal) {
    return (
      <div className="stage">
        <div className="processing-overlay" style={{ pointerEvents: 'auto' }}>
          <div className="processing-text">Cannot start</div>
          <div className="processing-sub">{fatal}</div>
        </div>
      </div>
    );
  }

  // Backend info still loading
  if (!backendInfo) {
    return (
      <div className="stage">
        <div className="processing-overlay">
          <div className="spinner" />
          <div className="processing-text">Connecting to AI…</div>
        </div>
      </div>
    );
  }

  // Backend reachable but no LIVE model loaded
  if (!backendInfo.live_available || !backendInfo.ws_url) {
    return (
      <div className="stage">
        <div className="processing-overlay" style={{ pointerEvents: 'auto' }}>
          <div className="processing-text">Live mode unavailable</div>
          <div className="processing-sub">
            {backendInfo.error || 'No LIVE model loaded on the vendor box. Start uvicorn and ensure DM-VTON or IDM-VTON is ready.'}
          </div>
        </div>
      </div>
    );
  }

  // ── Happy path: LIVE stream + dress picker ────────────────────────────
  const visibleList = activeGender === 'all'
    ? garments
    : garments.filter(g => g.gender === activeGender || g.gender === 'unisex');
  const presentGenders: Gender[] = Array.from(new Set(garments.map(g => g.gender)))
    .filter(g => g !== 'unisex') as Gender[];

  return (
    <div className="stage">
      {/* The actual LIVE stream — canvas that displays AI-rendered frames */}
      <LiveStream
        wsUrl={backendInfo.ws_url}
        garmentUrl={selectedGarment?.url || null}
        garmentCategory={selectedGarment?.category || 'dress'}
        liveMode={backendInfo.live_mode}
      />

      {/* Gender chips — top-right */}
      {presentGenders.length > 1 && (
        <div className="gender-chips">
          <button
            className={`g-chip ${activeGender === 'all' ? 'active' : ''}`}
            onClick={() => setActiveGender('all')}
          >All</button>
          {presentGenders.map(g => (
            <button
              key={g}
              className={`g-chip ${activeGender === g ? 'active' : ''}`}
              onClick={() => setActiveGender(g)}
            >{GENDER_EMOJI[g]} {GENDER_LABELS[g]}</button>
          ))}
        </div>
      )}

      {/* Dress picker — bottom strip */}
      <div className="picker">
        {visibleList.map(g => (
          <button
            key={g.id}
            className={`g-thumb ${selectedId === g.id ? 'selected' : ''}`}
            onClick={() => setSelectedId(g.id)}
            title={g.name}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={g.url} alt={g.name} loading="lazy" />
          </button>
        ))}
      </div>
    </div>
  );
}
