'use client';
// Customer page — photoreal snapshot try-on with IDM-VTON.
//
// Flow:
//   1. Live webcam preview (mirrored, full-screen). MediaPipe Pose runs
//      continuously, computing a "framing" check (face visible, shoulders
//      visible, hips visible, centered, full upper body).
//   2. User picks a dress from the bottom strip → enter 'preparing' phase.
//   3. While preparing, show framing instructions ("Step back", "Centre
//      yourself", "Show your face") until the user stays well-framed for
//      ~0.4 sec, then run a 3-2-1 countdown and snap.
//   4. Snapshot + garment posted to /api/tryon.
//   5. Job polled every 1.5s while AI runs (~165 sec on vendor box w/ CPU offload).
//   6. Result image fills the screen — customer can pick another dress.

import { useEffect, useRef, useState } from 'react';

type Category = 'top' | 'bottom' | 'dress';
type Gender = 'men' | 'women' | 'unisex';

declare global {
  interface Window {
    Pose?: any;
    Camera?: any;
  }
}

// What the framing checker tells the UI on every pose result.
interface Framing {
  ok: boolean;
  instruction: string;   // shown to the customer while framing is bad
}

// Run framing checks against MediaPipe pose landmarks (normalised image coords).
// Order matters — first failed check sets the instruction so we don't flicker.
function checkFraming(lm: any[] | null | undefined): Framing {
  if (!lm) return { ok: false, instruction: 'Step in front of the camera' };
  const vis = (i: number) => lm[i]?.visibility ?? 0;
  const x = (i: number) => lm[i]?.x ?? 0.5;
  const y = (i: number) => lm[i]?.y ?? 0.5;
  // Key landmarks: 0=nose, 11/12=shoulders, 23/24=hips
  if (vis(0) < 0.6)         return { ok: false, instruction: 'Step in front of the camera' };
  if (y(0) > 0.45)          return { ok: false, instruction: 'Step back a little' };
  if (y(0) < 0.05)          return { ok: false, instruction: 'Move down a bit' };
  if (vis(11) < 0.6 || vis(12) < 0.6) return { ok: false, instruction: 'Show both shoulders' };
  if (vis(23) < 0.6 || vis(24) < 0.6) return { ok: false, instruction: 'Step back — show your upper body' };
  // Nose x should be near centre. (Pose landmarks are on the un-mirrored
  // video frame, so "left" of the subject is high x. We just check distance
  // from centre 0.5 either way.)
  const noseDx = Math.abs(x(0) - 0.5);
  if (noseDx > 0.22)        return { ok: false, instruction: 'Centre yourself in the frame' };
  // Torso height check: shoulders→hips spans at least 20% of frame
  const shY = (y(11) + y(12)) / 2;
  const hpY = (y(23) + y(24)) / 2;
  const torsoH = hpY - shY;
  if (torsoH < 0.18)        return { ok: false, instruction: 'Step closer to the camera' };
  if (torsoH > 0.55)        return { ok: false, instruction: 'Step back a little' };
  return { ok: true, instruction: 'Ready — hold still' };
}

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

type Phase = 'live' | 'preparing' | 'capturing' | 'submitting' | 'processing' | 'result' | 'failed';

const POLL_INTERVAL_MS = 1500;
// Customer must stay well-framed for this many consecutive pose results
// before the countdown starts. ~30 fps × 0.4s ≈ 12 frames.
const FRAMING_HOLD_FRAMES = 12;
// Countdown shown before the snap actually fires
const COUNTDOWN_SECS = 3;

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

  // Pose + framing
  const [framing, setFraming] = useState<Framing>({ ok: false, instruction: 'Step in front of the camera' });
  const [countdown, setCountdown] = useState<number | null>(null);
  const framingRef = useRef<Framing>(framing);
  const phaseRef = useRef<Phase>('live');
  const garmentRef = useRef<Garment | null>(null);
  const okFramesRef = useRef(0);
  const countdownTimerRef = useRef<number | null>(null);

  const selectedGarment = garments.find(g => g.id === selectedId) || null;

  // Keep refs in sync with state so the pose-result callback always sees fresh values
  useEffect(() => { framingRef.current = framing; }, [framing]);
  useEffect(() => { phaseRef.current = phase; }, [phase]);
  useEffect(() => { garmentRef.current = selectedGarment; }, [selectedGarment]);

  // Camera + catalog + MediaPipe Pose setup
  useEffect(() => {
    let mounted = true;
    let stream: MediaStream | null = null;
    let mpCamera: any = null;

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

      // MediaPipe Pose runs continuously on the live video. We react to its
      // results to drive the framing instruction + auto-trigger snapshot
      // during the 'preparing' phase.
      const Pose = window.Pose;
      const Camera = window.Camera;
      if (!Pose || !Camera) {
        // MediaPipe scripts haven't loaded; we still let the user manually
        // tap a dress, just without framing guidance.
        console.warn('MediaPipe Pose not available — framing check disabled');
        return;
      }
      const pose = new Pose({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}` });
      pose.setOptions({
        modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false,
        minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      pose.onResults((r: any) => {
        const f = checkFraming(r.poseLandmarks);
        // Avoid setState churn when the instruction hasn't changed.
        if (framingRef.current.ok !== f.ok || framingRef.current.instruction !== f.instruction) {
          setFraming(f);
        }
        // Only react during 'preparing' phase
        if (phaseRef.current !== 'preparing') return;
        if (f.ok) {
          okFramesRef.current += 1;
          if (okFramesRef.current >= FRAMING_HOLD_FRAMES && countdownTimerRef.current === null) {
            startCountdown();
          }
        } else {
          okFramesRef.current = 0;
          if (countdownTimerRef.current !== null) {
            // Customer moved out of frame mid-countdown — abort it
            clearInterval(countdownTimerRef.current);
            countdownTimerRef.current = null;
            setCountdown(null);
          }
        }
      });

      mpCamera = new Camera(v, {
        onFrame: async () => { await pose.send({ image: v }); },
        width: v.videoWidth || 1280, height: v.videoHeight || 720,
      });
      mpCamera.start();
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
      try { mpCamera?.stop(); } catch {}
      stream?.getTracks().forEach(t => t.stop());
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, []);

  // Countdown ticker. After COUNTDOWN_SECS at 1Hz, fire captureAndSubmit().
  function startCountdown() {
    let n = COUNTDOWN_SECS;
    setCountdown(n);
    countdownTimerRef.current = window.setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(countdownTimerRef.current!);
        countdownTimerRef.current = null;
        setCountdown(null);
        const g = garmentRef.current;
        if (g) captureAndSubmit(g);
      } else {
        setCountdown(n);
      }
    }, 1000);
  }

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

  // Picker tap: enter 'preparing' phase. The pose-result handler will
  // auto-trigger captureAndSubmit() once framing has been ok for
  // FRAMING_HOLD_FRAMES consecutive results.
  function tryOnDress(g: Garment) {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    setJobId(null);
    setJobStatus(null);
    setSnapshotDataUrl(null);
    setSelectedId(g.id);
    setCountdown(null);
    okFramesRef.current = 0;
    setPhase('preparing');
    setStatus('frame yourself');
  }

  async function captureAndSubmit(g: Garment) {
    setPhase('capturing');
    setStatus('hold still…');
    await new Promise(r => setTimeout(r, 200));

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
      setStatus('generating your photo (~15-25 sec)…');
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

  const showLiveVideo = phase === 'live' || phase === 'preparing' || phase === 'capturing';
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

      {/* Framing instruction + countdown — visible only during 'preparing' */}
      {phase === 'preparing' && countdown === null && (
        <div className="framing-overlay">
          <div className="framing-instruction">{framing.instruction}</div>
          <div className="framing-sub">trying on {selectedGarment?.name || ''}</div>
        </div>
      )}
      {phase === 'preparing' && countdown !== null && (
        <div className="framing-overlay framing-countdown">
          <div className="framing-digit">{countdown}</div>
          <div className="framing-sub">hold still</div>
        </div>
      )}

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
