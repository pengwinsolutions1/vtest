// LocalBackend — talks to a vendor's on-prem GPU box over plain HTTP.
//
// The vendor's box must implement this contract:
//
//   POST /tryon/still   body: { selfie_url, garment_url }    → { prediction_id }
//   POST /tryon/video   body: { still_url }                  → { prediction_id }
//   GET  /predictions/{id}                                    → { status, output_url?, error? }
//
//   status enum: "pending" | "succeeded" | "failed"
//
// How the vendor implements it (ComfyUI workflows, raw PyTorch, whatever) is
// their problem. We only care about the HTTP surface. See
// docs/vendor-backend-contract.md for the spec.
//
// Auth: if VENDOR_API_KEY is set we pass it as Authorization: Bearer.
import type { AIBackend, GarmentCategory, PredictionPoll } from '../ai-backend';

export class LocalBackend implements AIBackend {
  constructor(private baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    const k = process.env.VENDOR_API_KEY;
    if (k) h['Authorization'] = `Bearer ${k}`;
    return h;
  }

  private async post(path: string, body: object): Promise<any> {
    const r = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    const text = await r.text();
    if (!r.ok) throw new Error(`vendor backend ${path} → HTTP ${r.status}: ${text}`);
    return text ? JSON.parse(text) : null;
  }

  private async get(path: string): Promise<any> {
    const r = await fetch(`${this.baseUrl}${path}`, { headers: this.headers() });
    const text = await r.text();
    if (!r.ok) throw new Error(`vendor backend ${path} → HTTP ${r.status}: ${text}`);
    return text ? JSON.parse(text) : null;
  }

  async kickoffStill(selfieUrl: string, garmentUrl: string, category: GarmentCategory): Promise<string> {
    const r = await this.post('/tryon/still', {
      selfie_url: selfieUrl,
      garment_url: garmentUrl,
      category,            // 'top' | 'bottom' | 'dress' — vendor maps to their model
    });
    if (!r?.prediction_id) throw new Error('vendor /tryon/still missing prediction_id in response');
    return String(r.prediction_id);
  }

  async kickoffVideo(stillUrl: string): Promise<string> {
    const r = await this.post('/tryon/video', { still_url: stillUrl });
    if (!r?.prediction_id) throw new Error('vendor /tryon/video missing prediction_id in response');
    return String(r.prediction_id);
  }

  async pollPrediction(predictionId: string): Promise<PredictionPoll> {
    const r = await this.get(`/predictions/${encodeURIComponent(predictionId)}`);
    const status = r?.status;
    if (status === 'succeeded') return { status: 'succeeded', output_url: String(r.output_url) };
    if (status === 'failed') return { status: 'failed', error: String(r.error || 'unknown') };
    if (status === 'pending') return { status: 'pending' };
    throw new Error(`vendor /predictions returned unknown status: ${status}`);
  }
}
