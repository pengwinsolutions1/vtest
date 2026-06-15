import './globals.css';
import type { Metadata, Viewport } from 'next';
import Script from 'next/script';

export const metadata: Metadata = {
  title: 'Virtual Try-On',
  description: 'See yourself wearing it, live.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: '#000',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* MediaPipe ships as UMD globals — Vite/Webpack tree-shake them away.
            Load via <Script> before React mounts so window.Pose / window.Camera
            exist for our client component. */}
        <Script
          src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
          strategy="beforeInteractive"
        />
        <Script
          src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"
          strategy="beforeInteractive"
        />
        {/* Selfie Segmentation gives us a per-pixel body mask. We use it to
            clip the garment overlay to the user's actual silhouette and to
            occlude arms that cross in front of the torso. */}
        <Script
          src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js"
          strategy="beforeInteractive"
        />
        {children}
      </body>
    </html>
  );
}
