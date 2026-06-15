'use client';
// Customer page — photoreal snapshot try-on with IDM-VTON.
//
// Flow:
//   1. Live webcam preview (mirrored, full-screen)
//   2. User picks a dress from the bottom strip
//   3. Snapshot captured from the webcam
//   4. Snapshot + garment posted to /api/tryon
//   5. Job polled every 1.5s while AI runs (~15-25s on vendor box)
//   6. Result image fills the screen — customer can browse another dress
//      to swap in a fresh try-on, or "Back to camera" to retake
//
// The LIVE WebSocket mode was tried first; it was real-time but the
// quality didn't reach what's expected. IDM-VTON gives photoreal stills
// — the closest to "real wearing" look that open AI currently produces.

import { useEffect, useRef, useState } from 'react';

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

interface JobStatus {
  id: string;
  status: 'queued' | 'still' | 'video' | 'succeeded' | 'failed';
  still_image_url?: string | null;
  video_url?: string | null;
  error?: string | null;
}

const GENDER_LABELS: Record<Gender, string> = { men: 'Men', women: 'Women', unisex: 'Unisex' };
const GENDER_EMOJI:  Record<Gender, string> = { men: '🧔', women: '👩', unisex: '🧑' };

type Phase = 'live' | 'capturing' | 'submitting' | 'processing' | 'result' | 'failed';

const POLL_INTERVAL_MS = 1500;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [garments, setGarments] = useState<Garment[]>([]);
  const [activeGender, setActiveGender] = useState<Gender | 'all'>('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('live');
  const [status, setStatus] = useState('starting…');
  const [fatal, setFatal] = useState<string | null>(null);

  const [snapshotDataUrl, setSnapshotDataUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);

  const selectedGarment = garments.find(g => g.id === selectedId) || null;

  // Camera + catalog setup
  useEffect(() => {
    let mounted = true;
    let stream: MediaStream | null = null;

    async function setup() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
      } catch (e: any) {
        setFatal(`Camera denied: ${e.name}. Open in a desktop browser with camera access.`);
        return;
      }
      if (!mounted) { stream.getTracks().forEach(t => t.stop()); return; }
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setStatus('tap a dress to try it on');
    }
    setup();

    fetch('/api/garments')
      .then(r => r.json())
      .then(d => {
        const list: Garment[] = d.garments || [];
        setGarments(list);
        if (list.length) setSelectedId(list[0].id);
      })
      .catch(e => setStatus(`catalog load failed: ${e.message}`));

    return () => {
      mounted = false;
      stream?.getTracks().forEach(t => t.stop());
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  // Take a snapshot from the live video — mirrored to match what the user sees.
  async function captureSnapshot(): Promise<{ blob: Blob; dataUrl: string } | null> {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    const w = v.videoWidth, h = v.videoHeight;
    const canvas = snapshotCanvasRef.current || document.createElement('canvas');
    snapshotCanvasRef.current = canvas;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.save();
    ctx.translate(w, 0); ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();
    const blob: Blob | null = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
    if (!blob) return null;
    return { blob, dataUrl: canvas.toDataURL('image/jpeg', 0.92) };
  }

  async function tryOnDress(g: Garment) {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setJobId(null);
    setJobStatus(null);
    setSelectedId(g.id);
    setPhase('capturing');
    setStatus('hold still…');
    await new Promise(r => setTimeout(r, 250));

    const snap = await captureSnapshot();
    if (!snap) {
      setStatus('snapshot failed — webcam not ready');
      setPhase('live');
      return;
    }
    setSnapshotDataUrl(snap.dataUrl);
    setPhase('submitting');
    setStatus('sending to AI…');

    const form = new FormData();
    form.append('selfie', snap.blob, 'selfie.jpg');
    form.append('garment_id', g.id);

    try {
      const r = await fetch('/api/tryon', { method: 'POST', body: form });
      const data = await r.json();
      if (!r.ok) {
        setStatus(`AI backend rejected: ${data.error}`);
        setPhase('failed');
        return;
      }
      setJobId(data.id);
      setPhase('processing');
      setStatus('generating your photo (~20 sec)…');
    } catch (e: any) {
      setStatus(`network: ${e.message}`);
      setPhase('failed');
    }
  }

  // Poll while a job is active. Stop on terminal states.
  useEffect(() => {
    if (!jobId) return;
    if (phase !== 'processing' && phase !== 'result') return;

    let stopped = false;
    async function poll() {
      try {
        const r = await fetch(`/api/jobs/${jobId}`);
        const data: JobStatus = await r.json();
        if (stopped) return;
        setJobStatus(data);
        if (data.status === 'succeeded') {
          setPhase('result');
          setStatus('done — looking good');
        } else if (data.status === 'failed') {
          setPhase('failed');
          setStatus(`failed: ${data.error || 'unknown error'}`);
        } else if (data.status === 'still') {
          setStatus('generating your photo (~20 sec)…');
        } else if (data.status === 'video' && data.still_image_url) {
          setPhase('result');
          setStatus('photo ready — adding motion…');
        }
      } catch (e: any) {
        console.warn('[poll]', e.message);
      }
    }

    poll();
    const id = window.setInterval(poll, POLL_INTERVAL_MS);
    pollTimerRef.current = id;
    return () => { stopped = true; clearInterval(id); pollTimerRef.current = null; };
  }, [jobId, phase]);

  function backToCamera() {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setJobId(null); setJobStatus(null); setSnapshotDataUrl(null);
    setPhase('live');
    setStatus('tap a dress to try it on');
  }

  const visibleList = activeGender === 'all'
    ? garments
    : garments.filter(g => g.gender === activeGender || g.gender === 'unisex');
  const presentGenders: Gender[] = Array.from(new Set(garments.map(g => g.gender)))
    .filter(g => g !== 'unisex') as Gender[];

  const showLiveVideo = phase === 'live' || phase === 'capturing';
  const showCaptured  = phase === 'submitting' || phase === 'processing';
  const showResult    = phase === 'result';

  return (
    <div className="stage">
      <video
        ref={videoRef}
        playsInline muted autoPlay
        className={`stage-video ${showLiveVideo ? '' : 'hidden'}`}
      />

      {showCaptured && snapshotDataUrl && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={snapshotDataUrl} alt="" className="stage-snapshot" />
      )}

      {showResult && jobStatus?.video_url ? (
        <video src={jobStatus.video_url} autoPlay loop muted playsInline className="stage-result" />
      ) : showResult && jobStatus?.still_image_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={jobStatus.still_image_url} alt="" className="stage-result" />
      ) : null}

      {phase === 'capturing' && <div className="capture-flash" />}

      <div className="status-pill">{fatal || status}</div>

      {presentGenders.length > 1 && phase === 'live' && (
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

      {/* Processing overlay */}
      {(phase === 'submitting' || phase === 'processing') && (
        <div className="processing-overlay">
          <div className="spinner" />
          <div className="processing-text">
            {phase === 'submitting' ? 'Uploading…' : 'Generating your photo (~20 sec)…'}
          </div>
          {selectedGarment && (
            <div className="processing-sub">Trying on {selectedGarment.name}</div>
          )}
        </div>
      )}

      {/* Result actions */}
      {showResult && (
        <button className="try-again-btn" onClick={backToCamera}>
          Back to camera
        </button>
      )}

      {/* Failed actions */}
      {phase === 'failed' && (
        <div className="processing-overlay" style={{ pointerEvents: 'auto' }}>
          <div className="processing-text">Try-on failed</div>
          <div className="processing-sub">{status}</div>
          <button className="try-again-btn try-again-btn-inline" onClick={backToCamera}>
            Back to camera
          </button>
        </div>
      )}

      <div className="picker">
        {visibleList.map(g => (
          <button
            key={g.id}
            className={`g-thumb ${selectedId === g.id ? 'selected' : ''}`}
            onClick={() => tryOnDress(g)}
            title={g.name}
            disabled={phase === 'capturing' || phase === 'submitting'}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={g.url} alt={g.name} loading="lazy" />
          </button>
        ))}
      </div>
    </div>
  );
}
