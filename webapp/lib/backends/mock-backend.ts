// MockBackend — returns canned results after fake delays. Used for UI
// development without spending money or needing a GPU.
//
// Mimics the real timing roughly so the progress UI feels right:
//   still:  ~10s pending then succeeded
//   video:  ~25s pending then succeeded
//
// Outputs are public sample assets so the result page can actually render.
// Swap these for your own samples if you want.
import { ulid } from 'ulid';
import type { AIBackend, GarmentCategory, PredictionPoll } from '../ai-backend';

interface MockPrediction {
  kind: 'still' | 'video';
  createdAt: number;
  delayMs: number;
}

// Pinned to globalThis so the store survives Next.js dev HMR and per-route
// module isolation. Without this, /api/tryon and /api/jobs each get their own
// fresh Map and predictions vanish between the kickoff and the first poll.
type G = typeof globalThis & { __tryon_mock_store__?: Map<string, MockPrediction> };
const g = globalThis as G;
const store = g.__tryon_mock_store__ ?? (g.__tryon_mock_store__ = new Map());

// Public samples for the mock outputs. Replace with your own if these go offline.
const SAMPLE_STILL = 'https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=800&h=1200&fit=crop';
const SAMPLE_VIDEO = 'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4';

export class MockBackend implements AIBackend {
  async kickoffStill(_selfieUrl: string, _garmentUrl: string, _category: GarmentCategory): Promise<string> {
    const id = `mock_still_${ulid()}`;
    store.set(id, { kind: 'still', createdAt: Date.now(), delayMs: 10_000 });
    return id;
  }

  async kickoffVideo(_stillUrl: string): Promise<string> {
    const id = `mock_video_${ulid()}`;
    store.set(id, { kind: 'video', createdAt: Date.now(), delayMs: 25_000 });
    return id;
  }

  async pollPrediction(predictionId: string): Promise<PredictionPoll> {
    const p = store.get(predictionId);
    if (!p) return { status: 'failed', error: `unknown mock prediction ${predictionId}` };
    if (Date.now() - p.createdAt < p.delayMs) return { status: 'pending' };
    return {
      status: 'succeeded',
      output_url: p.kind === 'still' ? SAMPLE_STILL : SAMPLE_VIDEO,
    };
  }
}
