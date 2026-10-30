import {test,expect} from '@playwright/test';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';

test('ensemble trial: real scoring, comparison, follow-up, metrics and known limitations',async({page,request})=>{
  const output=resolve('../../project_records/reports/figures',`Ensemble-${new Date().toISOString().replace(/[:.]/g,'-')}`);
  mkdirSync(output);
  const evidence={scope:'user_authorized_ensemble_trial',steps:[],human_acceptance:'PENDING'};
  const old=id=>resolve(`../../data/external20/images/images/${id}.png`);
  const fresh=id=>resolve(`../../data/image_eval_dataset_v2_20_anh/images/${id}.png`);
  const save=async name=>{await page.screenshot({path:resolve(output,`${name}.png`),fullPage:true});evidence.steps.push(name);};
  const start=async()=>{await page.goto('/');await expect(page.locator('.notice')).toContainText('ensemble thử nghiệm');};
  const score=async(prompt,file,name)=>{
    await start();await page.getByLabel('Prompt dùng để đối chiếu').fill(prompt);
    await page.getByLabel('Chọn ảnh').setInputFiles(file);
    const response=page.waitForResponse(r=>r.url().endsWith('/api/chat')&&r.status()===200);
    await page.getByRole('button',{name:'Chấm ảnh',exact:true}).click();
    const body=await (await response).json();
    await expect(page.locator('.result')).toHaveCount(1);
    await expect(page.locator('.metadata')).toContainText(evidence.model_version);
    evidence[name]={score:body.data.items[0].score,model_version:body.model_version};
    await save(name);return body.data.items[0].score;
  };
  try{
    const health=await (await request.get('/api/health')).json();
    expect(health.data.model_ready).toBe(true);expect(health.model_version).toMatch(/^ensemble25-/);
    evidence.model_version=health.model_version;
    await start();await save('01-home');
    await score('A red apple on a white plate.',old('01'),'02-apple');
    await start();await page.getByLabel('Prompt dùng để đối chiếu').fill('A red apple on a white plate.');
    await page.getByLabel('Chọn ảnh').setInputFiles([old('01'),old('20')]);
    const comparison=page.waitForResponse(r=>r.url().endsWith('/api/chat')&&r.status()===200);
    await page.getByRole('button',{name:'So sánh →',exact:true}).click();
    const body=await (await comparison).json();evidence.comparison=body.data.items;
    expect(body.data.items[0].score).toBeGreaterThan(body.data.items[1].score);
    await expect(page.locator('.result')).toHaveCount(2);await save('03-compare');
    const second=await page.locator('.result strong').nth(1).innerText();
    await page.getByLabel('Hỏi tiếp').fill('ảnh thứ hai thì sao');await page.getByRole('button',{name:'Gửi',exact:true}).click();
    await expect(page.locator('.result')).toHaveCount(3);await expect(page.locator('.result strong').last()).toHaveText(second);await save('04-follow-up');
    const mismatch=await score('A red bicycle parked in front of a white house.',old('20'),'05-car-bicycle');
    expect(mismatch).toBeCloseTo(1.322497,4);
    await start();await page.getByRole('button',{name:'Xem biểu đồ ↗'}).click();
    await expect(page.getByRole('row').filter({hasText:'ensemble25'})).toContainText('0.758');await save('06-metrics');
    const a=await score('Exactly two oranges on a blue ceramic plate. No other fruit. Overhead photograph.',fresh('05'),'07-two-oranges');
    const b=await score('Exactly three oranges on a blue ceramic plate. No other fruit. Overhead photograph.',fresh('05'),'08-known-counting-error');
    evidence.known_counting_limitation={correct:a,wrong:b,still_present:b>a};
    await start();await page.getByLabel('Chọn ảnh').setInputFiles(old('01'));
    await page.getByRole('button',{name:'Chấm ảnh',exact:true}).click();
    await expect(page.getByRole('alert')).toContainText('Nhập prompt');await save('09-empty-prompt');
    evidence.status='PASS_RUNTIME_WITH_KNOWN_MODEL_LIMITATIONS';
  }catch(e){evidence.status='FAIL';evidence.error=String(e);throw e;}
  finally{
    writeFileSync(resolve(output,'evidence.json'),JSON.stringify(evidence,null,2));
    writeFileSync(resolve(output,'index.html'),`<meta charset="utf-8"><title>Ensemble demo</title><h1>Ensemble thử nghiệm — chưa nghiệm thu release</h1>${evidence.steps.map(n=>`<h2>${n}</h2><img style="width:100%;max-width:1200px" src="${n}.png">`).join('')}`);
    console.log(output);
  }
});
