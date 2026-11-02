/* Exercise the generated offline HTML in an existing local headless Chrome. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');

async function main() {
  const targets = await (await fetch('http://127.0.0.1:19223/json/list')).json();
  const target = targets.find(t => t.type === 'page');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  let id=0;
  const pending=new Map(), errors=[];
  ws.addEventListener('message',event=>{
    const message=JSON.parse(event.data);
    if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);
    if(message.id){const handler=pending.get(message.id);pending.delete(message.id);message.error?handler.reject(message.error):handler.resolve(message.result);}
  });
  const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
  const evaluate=async expression=>{
    const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  try {
    await send('Runtime.enable');
    await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1120,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'report.html')).href});
    for(let attempt=0;attempt<50;attempt++){
      if(await evaluate("document.readyState==='complete' && !!document.getElementById('report-data') && document.getElementById('commit-count').textContent.includes('260')"))break;
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    const initial=await evaluate(`({cards:document.querySelectorAll('.perf-card').length,visible:document.querySelectorAll('.perf-card:not([hidden])').length,commit:$('commit-count').textContent,baseline:$('baseline-count').textContent,files:$('file-count').textContent,overflow:document.documentElement.scrollWidth>innerWidth,anchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(a.getAttribute('href').slice(1))).map(a=>a.getAttribute('href'))})`);
    assert.equal(initial.cards,71);assert.equal(initial.visible,71);assert.match(initial.commit,/260/);assert.match(initial.baseline,/3/);assert.match(initial.files,/1094/);assert.equal(initial.overflow,false);assert.deepEqual(initial.anchors,[]);
    const set=async(id,value)=>evaluate(`document.getElementById('${id}').value=${JSON.stringify(value)};document.getElementById('${id}').dispatchEvent(new Event('input'));true`);
    const count=()=>evaluate("document.querySelectorAll('.perf-card:not([hidden])').length");
    await set('perf-kind','existing');assert.equal(await count(),2);
    await set('perf-kind','withdrawn');assert.equal(await count(),5);
    await set('perf-search','unmatched_report_test_abc');assert.equal(await count(),0);
    await evaluate("$('perf-reset').click();true");assert.equal(await count(),71);
    await set('perf-evidence','measured');assert.equal(await count(),22);
    await evaluate("$('perf-reset').click();true");
    await set('commit-dedup','补丁等价');assert.match(await evaluate("$('commit-count').textContent"),/匹配 3/);
    await set('commit-dedup','未匹配等价项');assert.match(await evaluate("$('commit-count').textContent"),/匹配 257/);
    await set('commit-dedup','');await set('commit-perf','yes');assert.match(await evaluate("$('commit-count').textContent"),/匹配 71/);
    await set('commit-perf','');
    await evaluate("$('commit-next').click();true");assert.match(await evaluate("$('commit-page').textContent"),/第 2/);
    await evaluate("$('commit-prev').click();true");
    await set('baseline-dedup','补丁等价');assert.equal(await evaluate("$('baseline-body').children.length"),3);
    await set('baseline-dedup','');await set('file-search','qsfa');assert.ok(await evaluate("$('file-body').textContent.includes('qsfa')"));await set('file-search','');
    await evaluate("window.dispatchEvent(new Event('beforeprint'));true");
    assert.equal(await evaluate("$('commit-body').children.length"),260);assert.equal(await evaluate("$('file-body').children.length"),1094);
    await evaluate("window.dispatchEvent(new Event('afterprint'));true");
    const csvRows=await evaluate("Object.fromEntries(Object.entries(DATA.exports).map(([k,v])=>[k,v.split('\\n').length-1]))");
    assert.ok(csvRows['commits.csv']>=261);assert.ok(csvRows['files.csv']>=1095);
    const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(__dirname,'preview-desktop.png'),Buffer.from(desktop.data,'base64'));
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth'),false);
    const mobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(__dirname,'preview-mobile.png'),Buffer.from(mobile.data,'base64'));
    await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1120,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'../index.html')).href});
    for(let attempt=0;attempt<50;attempt++){
      if(await evaluate("document.readyState==='complete' && !!document.getElementById('matrix-count') && document.getElementById('matrix-count').textContent.includes('15')"))break;
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    assert.equal(await evaluate("document.querySelectorAll('.matrix-row').length"),15);
    assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth'),false);
    await set('matrix-search','19550');assert.equal(await evaluate("document.querySelectorAll('.matrix-row:not([hidden])').length"),1);
    await set('matrix-search','unmatched_report_test_abc');assert.equal(await evaluate("document.querySelectorAll('.matrix-row:not([hidden])').length"),0);
    await evaluate("document.getElementById('matrix-reset').click();true");
    const overviewDesktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(__dirname,'../preview-evolution-desktop.png'),Buffer.from(overviewDesktop.data,'base64'));
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate('document.documentElement.scrollWidth>innerWidth'),false);
    const overviewMobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(__dirname,'../preview-evolution-mobile.png'),Buffer.from(overviewMobile.data,'base64'));
    assert.deepEqual(errors,[]);
    const result={passed:true,initial,csvRows,checked:['offline load','no runtime exceptions','filtering','dedup counts','performance inventory','pagination','empty results','baseline mapping','file search','print expansion','CSV payload','desktop and mobile overflow','internal anchors','overview load','matrix filtering','overview desktop and mobile']};
    fs.writeFileSync(path.join(__dirname,'validation.json'),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result,null,2));
  } finally {ws.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
