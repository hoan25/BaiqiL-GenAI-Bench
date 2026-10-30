import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./tests-demo', workers:1, timeout:180000,
  use:{baseURL:'http://127.0.0.1:5173', browserName:'chromium',
       viewport:{width:1440,height:1000}, headless:false, launchOptions:{slowMo:150}},
});
