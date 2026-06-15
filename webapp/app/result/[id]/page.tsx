'use client';
// Result page — polls /api/jobs/[id] every 2 seconds until the job
// reaches a terminal state, then plays the video.
//
// The two-step pipeline gives us four UI states:
//   queued     — waiting for Replicate worker to pick up still-step
//   still      — IDM-VTON running (~15–30s)
//   video      — Wan 2.1 i2v running (~60–120s) — by now we can show the still
//   succeeded  — final MP4 ready
//   failed     — show error
import { useEffect, useState, use } from 'react';
import Link from 'next/link';

interface JobView {
  id: string;
  status: 'queued' | 'still' | 'video' | 'succeeded' | 'failed';
  still_image_url: string | null;
  video_url: string | null;
  error: string | null;
}

export default function ResultPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [job, setJob] = useState<JobView | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      try {
        const r = await fetch(`/api/jobs/${id}`, { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data: JobView = await r.json();
        if (stop) return;
        setJob(data);
        if (data.status !== 'succeeded' && data.status !== 'failed') {
          timer = setTimeout(tick, 2000);
        }
      } catch (e: any) {
        if (stop) return;
        setPollError(e.message);
        timer = setTimeout(tick, 5000); // back off on transient failure
      }
    }
    tick();
    return () => { stop = true; if (timer) clearTimeout(timer); };
  }, [id]);

  if (!job) {
    return (
      <div className="app">
        <div className="progress">
          <div className="spinner" />
          <h2>Connecting…</h2>
          {pollError && <div className="stage">{pollError}</div>}
        </div>
      </div>
    );
  }

  if (job.status === 'failed') {
    return (
      <div className="app">
        <div className="hero">
          <h1>Something went wrong</h1>
          <p>{job.error || 'Generation failed. Try again with different photos.'}</p>
        </div>
        <Link href="/" className="cta" style={{ textAlign: 'center', textDecoration: 'none', display: 'block' }}>
          Try again
        </Link>
      </div>
    );
  }

  if (job.status === 'succeeded' && job.video_url) {
    return (
      <div className="app">
        <div className="result">
          <h2>Here&apos;s you in it.</h2>
          <video src={job.video_url} controls autoPlay loop playsInline muted />
          <div className="actions">
            <a href={job.video_url} download className="cta" style={{ textDecoration: 'none', textAlign: 'center', marginTop: 0 }}>
              Download
            </a>
          </div>
          <div className="actions" style={{ marginTop: 10 }}>
            <Link href="/" style={{ flex: 1 }}>
              <button style={{ width: '100%' }}>Try another</button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // queued / still / video — show progress + preview-of-still once we have it
  const steps = [
    { key: 'queued', label: 'Queued' },
    { key: 'still', label: 'Generating try-on image' },
    { key: 'video', label: 'Animating it into video' },
    { key: 'succeeded', label: 'Done' },
  ];
  const order = ['queued', 'still', 'video', 'succeeded'];
  const currentIdx = order.indexOf(job.status);

  return (
    <div className="app">
      <div className="progress">
        <div className="spinner" />
        <h2>Working on your try-on…</h2>
        <div className="stage">
          {job.status === 'queued' && 'Waiting for a worker to pick it up.'}
          {job.status === 'still' && 'Step 1 of 2 — usually takes 15–30 seconds.'}
          {job.status === 'video' && 'Step 2 of 2 — animation takes 60–120 seconds.'}
        </div>
        <ul className="step-list">
          {steps.slice(0, 3).map((s, i) => (
            <li key={s.key} className={i < currentIdx ? 'done' : i === currentIdx ? 'active' : ''}>
              <span className="dot" />
              {s.label}
            </li>
          ))}
        </ul>

        {job.still_image_url && job.status === 'video' && (
          <div style={{ marginTop: 32 }}>
            <div className="stage" style={{ marginBottom: 12 }}>Try-on still ready — animating now:</div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={job.still_image_url} alt="try-on still" style={{ width: '100%', borderRadius: 12 }} />
          </div>
        )}
      </div>
    </div>
  );
}
