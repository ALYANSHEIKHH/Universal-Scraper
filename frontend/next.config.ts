/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  // Add if you need static export
  // output: 'export',
  // trailingSlash: true,
}

module.exports = nextConfig