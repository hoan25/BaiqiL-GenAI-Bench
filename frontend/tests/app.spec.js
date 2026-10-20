import {test, expect} from '@playwright/test';
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=', 'base64');
test.beforeEach(async ({page}) => {
  await page.route('**/api/**', async route => {
    const name = new URL(route.request().url()).pathname.split('/').pop();
    let type = name, data = {};
    if (name === 'health') data = {model_ready: true, scope: 'test_fixture'};
    if (name === 'sessions') data = {session_id: 'test-session'};
    if (name === 'uploads') data = {image_ids: ['a', 'b']};
    if (name === 'chat') {
      const input = route.request().postDataJSON();
      type = input.action === 'compare' ? 'ranking' : 'score';
      data = {items: Array.from({length: input.action === 'compare' ? 2 : 1}, (_, i) => ({index: i, rank: i + 1, score: 4 - i, cosine: 0.3}))};
    }
    if (name === 'metrics') {type = 'metrics'; data = {scope: 'test_fixture', split: 'val', models: {mean: {mae: 1, rmse: 1.2, spearman: null}}};}
    await route.fulfill({json: {type, data, model_version: 'mock-test-only', elapsed_ms: 5, warnings: []}});
  });
  await page.goto('/');
});
test('one image scoring', async ({page}) => {
  await page.getByLabel('Prompt dùng để đối chiếu').fill('a red square');
  await page.getByLabel('Chọn ảnh').setInputFiles({name: 'red.png', mimeType: 'image/png', buffer: png});
  await page.getByRole('button', {name: 'Chấm ảnh', exact: true}).click();
  await expect(page.locator('.result')).toHaveCount(1);
  await expect(page.locator('.result')).toContainText('4.00');
  await expect(page.locator('.metadata')).toContainText('mock-test-only');
});
test('comparison keeps order', async ({page}) => {
  await page.getByLabel('Prompt dùng để đối chiếu').fill('two images');
  await page.getByLabel('Chọn ảnh').setInputFiles(['a', 'b'].map(n => ({name: n + '.png', mimeType: 'image/png', buffer: png})));
  await page.getByRole('button', {name: 'So sánh →', exact: true}).click();
  await expect(page.locator('.result')).toHaveCount(2);
  await expect(page.locator('.result').nth(1)).toContainText('Ảnh 2 · Hạng 2');
});
test('dashboard displays undefined as NA', async ({page}) => {
  await page.getByRole('button', {name: 'Xem biểu đồ ↗'}).click();
  await expect(page.getByRole('cell', {name: 'N/A'})).toBeVisible();
});
test('invalid file and missing model are clear', async ({page}) => {
  await page.getByLabel('Chọn ảnh').setInputFiles({name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('bad')});
  await expect(page.getByRole('alert')).toContainText('JPG');
  await page.route('**/api/metrics', route => route.fulfill({status: 503, json: {data: {message: 'Chưa có mô hình được cấu hình.'}}}));
  await page.getByRole('button', {name: 'Xem biểu đồ ↗'}).click();
  await expect(page.getByRole('alert')).toContainText('Chưa có mô hình');
});
test('layout stays within viewport', async ({page}, testInfo) => {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.screenshot({path: `../reports/figures/G5-${testInfo.project.name}.png`, fullPage: true});
});
