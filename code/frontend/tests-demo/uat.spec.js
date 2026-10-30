import {test,expect} from '@playwright/test';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';

test('demo local: real images, follow-up, metrics and validation errors',async({page,request})=>{
  const stamp=new Date().toISOString().replace(/[:.]/g,'-');
  const output=resolve('../../project_records/reports/figures',`UAT-${stamp}`);
  mkdirSync(output,{recursive:false});
  const evidence={operator:'Codex browser automation, user authorized',human_acceptance:'PENDING',steps:[],responses:[]};
  const img=id=>resolve(`../../data/external20/images/images/${id}.png`);
  const prompt='A red apple on a white plate.';
  const save=async(name)=>{
    await page.evaluate(()=>document.fonts.ready);
    await page.screenshot({path:resolve(output,`${name}.png`),fullPage:true,animations:'disabled'});
    evidence.steps.push({name,status:'PASS',screenshot:`${name}.png`});
  };
  const start=async()=>{await page.goto('/');await expect(page.locator('.notice')).toContainText('validation GenAI-Bench');};
  page.on('response',async response=>{
    if(response.url().endsWith('/api/chat')&&response.status()===200){
      const body=await response.json();
      if(body.data?.items)evidence.responses.push({type:body.type,model_version:body.model_version,items:body.data.items});
    }
  });
  try{
    const health=await (await request.get('/api/health')).json();
    expect(health.data.model_ready).toBe(true);
    expect(health.model_version).toBe('mlp-b8dba09efa0e');
    evidence.model_version=health.model_version;
    await start();await save('01-home');
    await page.getByLabel('Prompt dùng để đối chiếu').fill(prompt);
    await page.getByLabel('Chọn ảnh').setInputFiles(img('01'));
    await page.getByRole('button',{name:'Chấm ảnh',exact:true}).click();
    await expect(page.locator('.result')).toHaveCount(1);
    await expect(page.locator('.metadata')).toContainText(health.model_version);
    await save('02-single-apple');
    await start();
    await page.getByLabel('Prompt dùng để đối chiếu').fill(prompt);
    await page.getByLabel('Chọn ảnh').setInputFiles([img('01'),img('20')]);
    await page.getByRole('button',{name:'So sánh →',exact:true}).click();
    await expect(page.locator('.result')).toHaveCount(2);
    const secondScore=await page.locator('.result strong').nth(1).innerText();
    await save('03-compare-same-prompt');
    await page.getByLabel('Hỏi tiếp').fill('ảnh thứ hai thì sao');
    await page.getByRole('button',{name:'Gửi',exact:true}).click();
    await expect(page.locator('.result')).toHaveCount(3);
    await expect(page.locator('.result strong').last()).toHaveText(secondScore);
    await save('04-follow-up');
    await start();await page.getByRole('button',{name:'Xem biểu đồ ↗'}).click();
    await expect(page.getByRole('row').filter({hasText:'mlp'})).toContainText('0.774');
    await expect(page.getByRole('row').filter({hasText:'CLIP cosine'})).toContainText('N/A');
    await expect(page.locator('.metrics')).toContainText('test holdout được báo cáo riêng');
    await save('05-validation-metrics');
    await start();await page.getByLabel('Chọn ảnh').setInputFiles(img('01'));
    await page.getByRole('button',{name:'Chấm ảnh',exact:true}).click();
    await expect(page.getByRole('alert')).toContainText('Nhập prompt');await save('06-empty-prompt');
    await start();await page.getByLabel('Chọn ảnh').setInputFiles({name:'not-image.txt',mimeType:'text/plain',buffer:Buffer.from('not an image')});
    await expect(page.getByRole('alert')).toContainText('JPG, PNG hoặc WebP');await save('07-invalid-file-type');
    await start();await page.getByLabel('Chọn ảnh').setInputFiles(Array(7).fill(img('01')));
    await expect(page.getByRole('alert')).toContainText('tối đa 6 ảnh');await save('08-too-many-images');
    await start();await page.getByLabel('Prompt dùng để đối chiếu').fill('A red bicycle parked in front of a white house.');
    await page.getByLabel('Chọn ảnh').setInputFiles(img('20'));
    await page.getByRole('button',{name:'Chấm ảnh',exact:true}).click();
    await expect(page.locator('.result')).toHaveCount(1);
    await expect(page.locator('.result strong')).toContainText('4.66');
    await save('09-known-model-error-car-vs-bicycle');
    evidence.status='PASS';
  }catch(error){evidence.status='FAIL';evidence.error=String(error);await page.screenshot({path:resolve(output,'failure.png'),fullPage:true});throw error;}
  finally{
    writeFileSync(resolve(output,'evidence.json'),JSON.stringify(evidence,null,2));
    console.log(`Demo evidence: ${output}`);
  }
});
