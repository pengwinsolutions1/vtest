'use client';
// LiveStream — the LIVE try-on UX.
//
// Captures webcam frames at ~10-15 fps, sends each as a JPEG over WebSocket
// to the vendor-box's /ws/live, displays the rendered frames the server
// returns on a <canvas>. The server runs DM-VTON (~50ms/frame, ~15-20fps
// perceived) or IDM-VTON low-step fallback (~3-5s/frame, quasi-live).
//
// Backpressure: we don't send a new frame until the server replies to the
// previous one. This matches the server's "drop frames when busy" policy
// and gives us the natural fps the backend can sustain.
//
// Garment switching: client sends a JSON text message {action: "set_garment",
// url, category} when the user picks a dress.

import { useEffect, useRef, useState } from 'react';

interface Props {
  wsUrl: string;
  garmentUrl: string | null;
  garmentCategory: 'top' | 'bottom' | 'dress';
  liveMode: 'dm_vton' | 'idm_vton_lowstep' | 'unavailable';
}

type Phase = 'connecting' | 'awaiting_garment' | 'streaming' | 'disconnected' | 'failed';

export default function LiveStream({ wsUrl, garmentUrl, garmentCategory, liveMode }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sendCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // Tracks whether a frame is in flight to the server (backpressure)
  const inflightRef = useRef(false);
  // Latest garment props (avoid stale closures in the capture loop)
  const garmentRef = useRef({ url: garmentUrl, category: garmentCategory });

  const [phase, setPhase] = useState<Phase>('connecting');
  const [status, setStatus] = useState('connecting to AI…');
  const [mode, setMode] = useState<string>('');

  // Keep ref in sync with props
  useEffect(() => {
    garmentRef.current = { url: garmentUrl, category: garmentCategory };
  }, [garmentUrl, garmentCategory]);

  // Open webcam + WebSocket on mount
  useEffect(() => {
    let mounted = true;
    let stream: MediaStream | null = null;
    let captureTimer: number | null = null;

    async function setup() {
      // 1. Webcam
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
      } catch (e: any) {
        setStatus(`camera denied: ${e.name}`);
        setPhase('failed');
        return;
      }
      if (!mounted) { stream.getTracks().forEach(t => t.stop()); return; }
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();

      // 2. WebSocket
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setPhase('awaiting_garment');
        setStatus('connected — pick a dress to start');
        // If a garment was already selected when we connected, tell the server
        if (garmentRef.current.url) sendSetGarment(garmentRef.current.url, garmentRef.current.category);
      };

      ws.onmessage = (evt) => {
        if (typeof evt.data === 'string') {
          // Text message from server (status info)
          try {
            const obj = JSON.parse(evt.data);
            if (obj.error) {
              setStatus(`server: ${obj.error}`);
              setPhase('failed');
            } else if (obj.mode) {
              setMode(obj.mode);
              setStatus(obj.mode === 'dm_vton'
                ? 'live mode (real-time)'
                : 'live mode (slow, ~3-5s/frame)');
            }
            // Whatever the message, server is now ready for another frame
            inflightRef.current = false;
          } catch { /* ignore */ }
          return;
        }
        // Binary message = rendered frame (JPEG)
        inflightRef.current = false;
        const blob = new Blob([evt.data as ArrayBuffer], { type: 'image/jpeg' });
        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;
          if (canvas.width !== img.width || canvas.height !== img.height) {
            canvas.width = img.width;
            canvas.height = img.height;
          }
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(img.src);
        };
        img.src = URL.createObjectURL(blob);
        if (phase !== 'streaming') {
          setPhase('streaming');
          setStatus(`streaming (${mode || 'live'})`);
        }
      };

      ws.onerror = () => {
        setStatus('WebSocket error');
        setPhase('failed');
      };

      ws.onclose = () => {
        setPhase('disconnected');
        setStatus('disconnected from AI');
      };

      // 3. Capture loop — try to send a frame every 80ms (target ~12fps),
      // but only if (a) garment is selected and (b) no frame in flight
      captureTimer = window.setInterval(() => {
        if (inflightRef.current) return;
        if (!garmentRef.current.url) return;
        if (ws.readyState !== WebSocket.OPEN) return;
        sendFrame(v, ws);
      }, 80);
    }

    function sendFrame(video: HTMLVideoElement, ws: WebSocket) {
      if (!video.videoWidth) return;
      const c = sendCanvasRef.current || (sendCanvasRef.current = document.createElement('canvas'));
      // DM-VTON's native is 192x256. Sending bigger wastes bandwidth.
      // Downsize aggressively + mirror so the server gets a selfie-view frame.
      const W = 256, H = 192;
      c.width = W; c.height = H;
      const ctx = c.getContext('2d')!;
      ctx.save();
      ctx.translate(W, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, W, H);
      ctx.restore();
      c.toBlob(async (blob) => {
        if (!blob) return;
        try {
          const buf = await blob.arrayBuffer();
          ws.send(buf);
          inflightRef.current = true;
        } catch { /* socket may have closed mid-send */ }
      }, 'image/jpeg', 0.75);
    }

    function sendSetGarment(url: string, category: 'top' | 'bottom' | 'dress') {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ action: 'set_garment', url, category }));
    }

    setup();

    return () => {
      mounted = false;
      if (captureTimer) clearInterval(captureTimer);
      try { wsRef.current?.close(); } catch {}
      stream?.getTracks().forEach(t => t.stop());
    };
  }, [wsUrl]);

  // When garment changes mid-stream, push it to the server
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!garmentUrl) return;
    ws.send(JSON.stringify({ action: 'set_garment', url: garmentUrl, category: garmentCategory }));
  }, [garmentUrl, garmentCategory]);

  // Initial label uses the prop directly; once the server replies, `mode`
  // state takes over and reflects the server's actual reported mode.
  const liveLabel = mode === 'dm_vton' || liveMode === 'dm_vton'
    ? 'LIVE'
    : 'LIVE (slow)';

  return (
    <>
      {/* Hidden video element feeds the send-canvas; not displayed directly */}
      <video ref={videoRef} playsInline muted autoPlay style={{ display: 'none' }} />
      {/* What the user sees: the rendered frame from the AI server */}
      <canvas ref={canvasRef} className="live-canvas" />
      <div className="status-pill">
        <span className="live-dot" /> {liveLabel} · {status}
      </div>
    </>
  );
}
