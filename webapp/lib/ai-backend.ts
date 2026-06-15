// AIBackend abstraction. Three implementations live alongside this file:
//   - replicate-backend.ts  → calls Replicate's hosted models (per-call billing)
//   - local-backend.ts      → calls a vendor's on-prem GPU box over HTTP
//   - mock-backend.ts       → returns canned sample data after a fake delay
//                              (for UI dev without spending money or needing a GPU)
//
// Pick at boot via env var:  AI_BACKEND=mock|local|replicate
//
// All backends report progress through the same job state machine:
//   queued → still → video → succeeded   (or → failed at any step)
//
// The route layer doesn't know which backend it's talking to.

export type GarmentCategory = 'top' | 'bottom' | 'dress';

export interface PredictionPoll {
  status: 'pending' | 'succeeded' | 'failed';
  output_url?: string;
  error?: string;
}

export interface AIBackend {
  // Kick off the try-on still step. Returns an opaque prediction id to poll later.
  // selfieUrl and garmentUrl must be PUBLIC URLs the backend can reach.
  // category routes the model: 'top' → upper_body, 'bottom' → lower_body, 'dress' → dresses.
  kickoffStill(selfieUrl: string, garmentUrl: string, category: GarmentCategory): Promise<string>;

  // Kick off the still-to-video animation step.
  kickoffVideo(stillUrl: string): Promise<string>;

  // Check current state of a previously-kicked-off prediction.
  // Backends should make this idempotent and cheap to call repeatedly.
  pollPrediction(predictionId: string): Promise<PredictionPoll>;
}

let cached: AIBackend | null = null;

export async function getBackend(): Promise<AIBackend> {
  if (cached) return cached;
  const kind = (process.env.AI_BACKEND || 'mock').toLowerCase();
  switch (kind) {
    case 'mock': {
      const { MockBackend } = await import('./backends/mock-backend');
      cached = new MockBackend();
      break;
    }
    case 'replicate': {
      const { ReplicateBackend } = await import('./backends/replicate-backend');
      cached = new ReplicateBackend();
      break;
    }
    case 'local': {
      const { LocalBackend } = await import('./backends/local-backend');
      const base = process.env.LOCAL_AI_URL;
      if (!base) throw new Error('AI_BACKEND=local requires LOCAL_AI_URL');
      cached = new LocalBackend(base);
      break;
    }
    default:
      throw new Error(`unknown AI_BACKEND=${kind}; expected one of: mock | replicate | local`);
  }
  return cached;
}
