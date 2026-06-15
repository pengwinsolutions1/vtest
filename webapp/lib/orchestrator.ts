// Job orchestrator — the only code that knows about the state machine.
// Routes call kickoffJob() once at creation, then advanceJob() on every poll.
// advanceJob is idempotent and rate-limited (max one backend poll per job per 2s)
// so it's safe to call freely from the browser-driven /api/jobs/[id] endpoint.
import { updateJob, getJob, type Job, type GarmentCategory } from './db';
import { getBackend } from './ai-backend';

// Kicked off when a new job is created. Triggers the still-image step.
// `category` routes the backend to the correct try-on model variant.
export async function kickoffJob(job: Job, category: GarmentCategory): Promise<void> {
  const backend = await getBackend();
  const predId = await backend.kickoffStill(job.selfie_url, job.garment_url, category);
  updateJob(job.id, { status: 'still', still_prediction_id: predId });
}

const lastChecked = new Map<string, number>();
const CHECK_INTERVAL_MS = 2000;

export async function advanceJob(jobId: string): Promise<void> {
  const job = getJob(jobId);
  if (!job) return;
  if (job.status === 'succeeded' || job.status === 'failed') return;

  const now = Date.now();
  const last = lastChecked.get(jobId) || 0;
  if (now - last < CHECK_INTERVAL_MS) return;
  lastChecked.set(jobId, now);

  const backend = await getBackend();
  const predId = job.status === 'still' ? job.still_prediction_id
              : job.status === 'video' ? job.video_prediction_id
              : null;
  if (!predId) return;

  let poll;
  try {
    poll = await backend.pollPrediction(predId);
  } catch (e: any) {
    // Transient network failure — next poll retries.
    console.warn(`[advance] backend poll failed for ${predId}: ${e.message}`);
    return;
  }

  if (poll.status === 'pending') return;
  if (poll.status === 'failed') {
    updateJob(job.id, { status: 'failed', error: poll.error || 'unknown' });
    return;
  }

  // succeeded — advance to the next step or finish
  if (job.status === 'still') {
    updateJob(job.id, { still_image_url: poll.output_url || null });
    const updated = getJob(job.id)!;
    try {
      const videoPredId = await backend.kickoffVideo(updated.still_image_url!);
      updateJob(job.id, { status: 'video', video_prediction_id: videoPredId });
    } catch (e: any) {
      // Video step unavailable (e.g. Wan 2.1 not loaded on the vendor box).
      // The still image is the result — finish successfully on it.
      console.warn(`[advance] video step unavailable (${e.message}) — finishing on still only`);
      updateJob(job.id, { status: 'succeeded' });
    }
  } else if (job.status === 'video') {
    updateJob(job.id, { status: 'succeeded', video_url: poll.output_url || null });
  }
}
