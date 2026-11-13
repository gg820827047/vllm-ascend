const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
(async()=>{
  const targets=await(await fetch('http://127.0.0.1:19225/json/list')).json();
  const target=targets.find(t=>t.type==='page');
  const ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
  let seq=0;const pending=new Map(),errors=[];
  ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  try{
    await send('Runtime.enable');await send('Page.enable');
    await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1000,deviceScaleFactor:1,mobile:false});
    await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'report.html')).href});
    for(let i=0;i<120;i++){if(await evaluate('window.reportReady===true'))break;await new Promise(r=>setTimeout(r,100))}
    assert.equal(await evaluate('window.reportReady'),true);
    await evaluate("document.documentElement.style.scrollBehavior='auto';true");
    const facts=await evaluate(`({ranks:D.ranks.length,windows:D.ranksteps.length,exact:D.exact_shapes.length,rectangles:hitRects.length,heads:document.querySelectorAll('h2').length,nav:document.querySelectorAll('#nav a').length,overflow:document.documentElement.scrollWidth>innerWidth})`);
    assert.equal(facts.ranks,28);assert.equal(facts.windows,122);assert.equal(facts.exact,7);assert.ok(facts.rectangles>1000);assert.equal(facts.heads,facts.nav);assert.equal(facts.overflow,false);
    await evaluate(`$('op-filter').value='MegaMoe';renderOps();true`);
    assert.equal(await evaluate(`$('op-table').querySelectorAll('tbody tr').length`),1);
    await evaluate(`$('op-filter').value='';renderOps();$('phase-select').value='prep';resetView();true`);
    assert.ok(await evaluate(`hitRects.some(r=>r.info.name==='gloo:all_reduce')`));
    await evaluate(`$('rank-select').value='a5_bs12-dp0-tp0';populateSteps();$('step-select').value='2';$('phase-select').value='main';resetView();true`);
    assert.equal(await evaluate('timelineState.step'),2);
    assert.ok((await evaluate(`$('window-caption').textContent`)).includes('prefill'));
    const rect=await evaluate(`(()=>{const r=hitRects.filter(r=>r.info.type==='MegaMoe').sort((a,b)=>b.info.duration_us-a.info.duration_us)[0];return {x:r.x+Math.min(5,r.w/2),y:r.y+12,d:r.info.duration_us}})()`);
    assert.ok(rect.d>40000);
    await evaluate(`$('timeline').scrollIntoView({block:'center'});true`);
    const box=await evaluate(`(()=>{const b=$('timeline').getBoundingClientRect();return {x:b.left,y:b.top}})()`);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,x:box.x+rect.x,y:box.y+rect.y});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,x:box.x+rect.x,y:box.y+rect.y});
    assert.ok((await evaluate(`$('inspector').textContent`)).includes('csv_line'));
    await evaluate(`$('rank-select').value='a3_recent-dp0-tp0';populateSteps();$('phase-select').value='prep';resetView();true`);
    let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-timeline.png'),Buffer.from(shot.data,'base64'));
    await evaluate('window.scrollTo(0,0);true');
    shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-desktop.png'),Buffer.from(shot.data,'base64'));
    await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await evaluate('window.scrollTo(0,0);draw();true');
    await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
    const mobile=await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth})');assert.equal(mobile.overflow,false);assert.equal(mobile.width,390);
    shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-mobile.png'),Buffer.from(shot.data,'base64'));
    assert.deepEqual(errors,[]);
    const result={passed:true,facts,mobile,inspectedMegaDurationUs:rect.d,javascriptErrors:errors};fs.writeFileSync(path.join(__dirname,'browser_validation.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  }finally{ws.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
