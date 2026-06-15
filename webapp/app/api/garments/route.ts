// GET /api/garments
// Optional filters:  ?category=top|bottom|dress   ?gender=men|women|unisex
//
// Response shape per garment:
//   { id, name, category, gender, url, glb_url?, glb_status? }
// glb_url is only set when a ready .glb file exists in the catalog; otherwise
// the live AR falls back to a procedural mesh textured with the PNG.
import { NextRequest, NextResponse } from 'next/server';
import { listGarments } from '@/lib/db';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const category = req.nextUrl.searchParams.get('category');
  const gender = req.nextUrl.searchParams.get('gender');
  let garments = listGarments();
  if (category) garments = garments.filter(g => g.category === category);
  if (gender)   garments = garments.filter(g => g.gender === gender);

  const base = (process.env.PUBLIC_URL || 'http://localhost:3000').replace(/\/$/, '');
  return NextResponse.json({
    garments: garments.map(g => ({
      id: g.id,
      name: g.name,
      category: g.category,
      gender: g.gender,
      url: `${base}/uploads/garments/${g.filename}`,
      glb_url: g.glb_filename && g.glb_status === 'ready'
        ? `${base}/uploads/garments/${g.glb_filename}`
        : null,
      glb_status: g.glb_status,
    })),
  });
}
