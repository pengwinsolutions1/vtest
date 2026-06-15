/** @type {import('next').NextConfig} */
module.exports = {
  // better-sqlite3 is a native module — keep it out of the server bundle.
  serverExternalPackages: ['better-sqlite3'],
  // Allow Replicate-hosted result URLs to render in <video> tags without CORS hassle.
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: 'replicate.delivery' },
      { protocol: 'https', hostname: 'pbxt.replicate.delivery' },
    ],
  },
};
