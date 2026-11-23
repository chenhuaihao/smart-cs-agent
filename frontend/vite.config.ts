import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 开发态代理 /api -> FastAPI (127.0.0.1:8000)，显式关闭压缩并禁用响应缓冲以保证 SSE 逐帧透传
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // 对 SSE 请求透传 Accept-Encoding: identity，防止代理层因 gzip 聚合导致流式分片延迟
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq, req) => {
            if (req.url?.includes('/stream') || req.url?.includes('/approve')) {
              proxyReq.setHeader('Accept-Encoding', 'identity');
            }
          });
          proxy.on('proxyRes', (proxyRes) => {
            const ct = proxyRes.headers['content-type'];
            if (typeof ct === 'string' && ct.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache';
              proxyRes.headers['x-accel-buffering'] = 'no';
            }
          });
        },
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
