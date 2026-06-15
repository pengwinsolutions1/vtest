'use client';
// Customer page — photo-booth UX for diffusion-based virtual try-on.
//
// Flow:
//   1. Live webcam preview (mirrored, cover-fit). User browses garments in the
//      bottom strip.
//   2. User selects a garment, then taps "Try On". A still snapshot is taken
//      from the webcam, sent to POST /api/tryon with the garment_id.
//   3. We poll GET /api/jobs/[id]. The orchestrator runs IDM-VTON (still
//      photo, ~10–20s) then Wan 2.1 (i2v video, ~25–60s).
//   4. As soon as the still URL is available, we show it on screen. When the
//      video is available, we swap in the video element.
//   5. "Try Another" returns to live preview.
//
// Backends are pluggable via env: AI_BACKEND=mock|replicate|local. The webapp
// doesn't care which one is actually generating the image; it just talks to
// /api/tryon and /api/jobs. See lib/ai-backend.ts.

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

// UI state for the page. 'live' lets the customer browse; 'processing' shows
// the snapshot + progress; 'result' shows the AI image/video.
type Phase = 'live' | 'capturing' | 'submitting' | 'processing' | 'result' | 'failed';

const POLL_INTERVAL_MS = 1500;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [garments, setGarments] = useState<Garment[]>([]);
  const [activeGender, setActiveGender] = useState<Gender | 'all'>('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('live');
  const [status, setStatus] = useState('starting…');
  const [fatal, setFatal] = useState<string | null>(null);

  // Snapshot we sent to the backend, kept as a data URL so we can display it
  // during processing as a "you'll look like this with the garment" placeholder.
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

  // Take a snapshot from the live video, return a Blob (JPEG) and data URL.
  async function captureSnapshot(): Promise<{ blob: Blob; dataUrl: string } | null> {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;

    // Mirror so the snapshot matches what the user just saw on screen.
    const w = v.videoWidth, h = v.videoHeight;
    const canvas = snapshotCanvasRef.current || document.createElement('canvas');
    snapshotCanvasRef.current = canvas;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();

    const blob: Blob | null = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
    if (!blob) return null;
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    return { blob, dataUrl };
  }

  async function startTryOn(g?: Garment) {
    const target = g || selectedGarment;
    if (!target) return;
    setPhase('capturing');
    setStatus('hold still…');

    // Brief beat so the UI flash is visible
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
    form.append('garment_id', target.id);

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
      setStatus('generating your photo…');
    } catch (e: any) {
      setStatus(`network: ${e.message}`);
      setPhase('failed');
    }
  }

  // Poll the job once we have an id. Keep polling until succeeded/failed so
  // the video step completes — earlier version stopped on the still preview
  // and the customer never saw the video.
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
          setStatus('generating your photo (about 20 sec)…');
        } else if (data.status === 'video') {
          // Still is ready, video is being generated. Show the still as a
          // preview so the user isn't staring at a frozen snapshot, but keep
          // polling so the video replaces it when ready.
          if (data.still_image_url) {
            setPhase('result');
            setStatus('adding motion (about 30 sec)…');
          } else {
            setStatus('queued for motion generation…');
          }
        }
      } catch (e: any) {
        console.warn('[poll]', e.message);
      }
    }

    poll();
    const id = window.setInterval(poll, POLL_INTERVAL_MS);
    pollTimerRef.current = id;
    return () => {
      stopped = true;
      clearInterval(id);
      pollTimerRef.current = null;
    };
  }, [jobId, phase]);

  function tryAnother() {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setJobId(null);
    setJobStatus(null);
    setSnapshotDataUrl(null);
    setPhase('live');
    setStatus('pick a dress to try on');
  }

  // Tapping a dress thumbnail IS the try-on action. From any phase, picking a
  // different dress aborts the current job and starts a new try-on with the
  // fresh selection. No separate "Try On" button — keep it simple.
  function onPickDress(g: Garment) {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setJobId(null);
    setJobStatus(null);
    setSelectedId(g.id);
    // Defer to next tick so React commits selectedId before startTryOn reads it
    setTimeout(() => startTryOn(g), 0);
  }

  const visibleList = activeGender === 'all'
    ? garments
    : garments.filter(g => g.gender === activeGender || g.gender === 'unisex');
  const presentGenders: Gender[] = Array.from(new Set(garments.map(g => g.gender)))
    .filter(g => g !== 'unisex') as Gender[];

  const showLiveVideo = phase === 'live' || phase === 'capturing';
  const showCapturedSnapshot = phase === 'submitting' || phase === 'processing';
  const showResult = phase === 'result';

  return (
    <div className="stage">
      {/* Live video — mirrored, cover-fit. Hidden when we have a snapshot/result. */}
      <video
        ref={videoRef}
        playsInline muted autoPlay
        className={`stage-video ${showLiveVideo ? '' : 'hidden'}`}
      />

      {/* Snapshot (frozen frame) — shown during processing */}
      {showCapturedSnapshot && snapshotDataUrl && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={snapshotDataUrl} alt="" className="stage-snapshot" />
      )}

      {/* AI result — still or video */}
      {showResult && jobStatus?.video_url ? (
        <video
          src={jobStatus.video_url}
          autoPlay loop muted playsInline
          className="stage-result"
        />
      ) : showResult && jobStatus?.still_image_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={jobStatus.still_image_url} alt="" className="stage-result" />
      ) : null}

      {/* Capture flash */}
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

      {/* Progress overlay — during processing */}
      {(phase === 'submitting' || phase === 'processing') && (
        <div className="processing-overlay">
          <div className="spinner" />
          <div className="processing-text">
            {phase === 'submitting' ? 'Uploading…' : (
              jobStatus?.status === 'still' ? 'Generating photo (about 20 sec)…' :
              jobStatus?.status === 'video' ? 'Adding motion (about 30 sec)…' :
              'Queued…'
            )}
          </div>
          {selectedGarment && (
            <div className="processing-sub">Trying on {selectedGarment.name}</div>
          )}
        </div>
      )}

      {/* Result actions — "Back to live" reverts to the camera preview */}
      {showResult && (
        <button className="try-again-btn" onClick={tryAnother}>
          Back to camera
        </button>
      )}

      {/* Failed actions */}
      {phase === 'failed' && (
        <div className="processing-overlay">
          <div className="processing-text">Try-on failed</div>
          <div className="processing-sub">{status}</div>
          <button className="try-again-btn try-again-btn-inline" onClick={tryAnother}>
            Try Again
          </button>
        </div>
      )}

      {/* Dress picker is always visible (live / processing / result). Picking
          a different dress aborts the current job and starts a new try-on. */}
      <div className="picker">
        {visibleList.map(g => (
          <button
            key={g.id}
            className={`g-thumb ${selectedId === g.id ? 'selected' : ''}`}
            onClick={() => onPickDress(g)}
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
