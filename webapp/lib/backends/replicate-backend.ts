// Replicate adapter — calls cuuupid/idm-vton for try-on stills and
// wavespeedai/wan-2.1-i2v-720p for image-to-video. Billed per-call.
//
// LICENSE NOTE: IDM-VTON is non-commercial-only per its model card. Fine for
// dev/demo; swap to a commercially-licensed model (fal.ai's fashn-tryon-v1)
// before launching a paid product.
import Replicate from 'replicate';
import type { AIBackend, GarmentCategory, PredictionPoll } from '../ai-backend';

// Our internal category names → IDM-VTON's `category` enum values.
const CATEGORY_MAP: Record<GarmentCategory, string> = {
  top:    'upper_body',
  bottom: 'lower_body',
  dress:  'dresses',
};

// Pinned version hashes — verified against Replicate's API. Update consciously.
const IDM_VTON_VERSION = '0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985';
const WAN_I2V_VERSION = '1f0a7fa066689a087b597a314f60ef74d1a720fa1fb9a7083487c4b01db3395f';

export class ReplicateBackend implements AIBackend {
  private client: Replicate;

  constructor() {
    if (!process.env.REPLICATE_API_TOKEN) {
      throw new Error('REPLICATE_API_TOKEN is not set');
    }
    this.client = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  }

  async kickoffStill(selfieUrl: string, garmentUrl: string, category: GarmentCategory): Promise<string> {
    const pred = await this.client.predictions.create({
      version: IDM_VTON_VERSION,
      input: {
        human_img: selfieUrl,
        garm_img: garmentUrl,
        garment_des: `a ${category} clothing item to try on the person`,
        category: CATEGORY_MAP[category],
        crop: false,
        seed: 42,
      },
    });
    return pred.id;
  }

  async kickoffVideo(stillUrl: string): Promise<string> {
    const pred = await this.client.predictions.create({
      version: WAN_I2V_VERSION,
      input: {
        image: stillUrl,
        prompt: 'a person standing naturally, slight body movement, breathing, gentle camera motion',
        negative_prompt: 'distorted, deformed, glitchy, low quality, watermark, extra limbs',
        aspect_ratio: '9:16',
        sample_guide_scale: 5,
        sample_shift: 3,
        sample_steps: 30,
        fast_mode: 'Balanced',
      },
    });
    return pred.id;
  }

  async pollPrediction(predictionId: string): Promise<PredictionPoll> {
    const pred = await this.client.predictions.get(predictionId);
    if (pred.status === 'succeeded') {
      const url = Array.isArray(pred.output) ? String(pred.output[0]) : String(pred.output);
      return { status: 'succeeded', output_url: url };
    }
    if (pred.status === 'failed' || pred.status === 'canceled') {
      return { status: 'failed', error: String(pred.error || pred.status) };
    }
    return { status: 'pending' };
  }
}
