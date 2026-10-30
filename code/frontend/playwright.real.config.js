import {defineConfig} from '@playwright/test';
import {resolve} from 'node:path';
const root = resolve('..');
export default defineConfig({
  testDir: './tests-real', timeout: 120000,
  use: {baseURL: 'http://127.0.0.1:5173', browserName: 'chromium'},
  webServer: [
    {command: `"${resolve(root, '../.venv/Scripts/python.exe')}" -X utf8 -m uvicorn backend.app.main:configured_app --factory --host 127.0.0.1 --port 8000`,
     cwd: root, url: 'http://127.0.0.1:8000/api/health', timeout: 120000, reuseExistingServer: false,
     env: {PROMPTALIGN_CLIP: resolve(root, '../.cache/models/clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),
           PROMPTALIGN_TRAINING: resolve(root, '../.cache/reviewed-training-20261020T174855207187Z/training-seed42'),
           PROMPTALIGN_ALLOW_DEVELOPMENT: '0', HF_HUB_OFFLINE: '1', OMP_NUM_THREADS: '4'}},
    {command: 'node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173', url: 'http://127.0.0.1:5173', reuseExistingServer: false}
  ]
});
