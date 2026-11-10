/* Validate the offline HTML using local Chrome's DevTools protocol. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
(async()=>{
 const targets=await(await fetch('http://127.0.0.1:19224/json/list')).json();
 const target=targets.find(t=>t.type==='page');
 const ws=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((res,rej)=>{ws.addEventListener('open',res,{once:true});ws.addEventListener('error',rej,{once:true});});
 let sequence=0;const pending=new Map(),errors=[];
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}});
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 try{
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(path.join(__dirname,'report.html')).href});
  for(let i=0;i<100;i++){if(await evaluate('window.reportReady===true'))break;await new Promise(r=>setTimeout(r,100));}
  assert.equal(await evaluate('window.reportReady'),true);
  await evaluate(`document.documentElement.style.scrollBehavior='auto';true`);
  const desktop=await evaluate(`({cards:document.querySelectorAll('.pr-card').length,overflow:document.documentElement.scrollWidth>innerWidth,brokenAnchors:[...document.querySelectorAll('aside a')].filter(a=>!document.getElementById(a.hash.slice(1))).length,timelineRows:D.timelines.length,layerRows:D.layers.length})`);
  assert.equal(desktop.cards,60);assert.equal(desktop.overflow,false);assert.equal(desktop.brokenAnchors,0);assert.equal(desktop.timelineRows,6);assert.equal(desktop.layerRows,240);
  const filtered=await evaluate(`$('pr-status').value='机制已验证';filterPR();document.querySelectorAll('.pr-card:not([hidden])').length`);assert.equal(filtered,2);
  const selected=await evaluate(`$('pr-status').value='';$('pr-search').value='19657';filterPR();document.querySelectorAll('.pr-card:not([hidden])').length`);assert.equal(selected,1);
  await evaluate(`$('pr-search').value='';filterPR();$('preset-wait').click();true`);
  const waiting=await evaluate(`({index:timelineIndex,rectangles:hitRects.length,mega:hitRects.filter(r=>r.info.name==='MegaMoe').reduce((a,b)=>a.duration>b.duration?a:b)})`);
  assert.equal(waiting.index,4);assert.ok(waiting.rectangles>0);assert.ok(waiting.mega.duration>17e6);
  const click=await evaluate(`(()=>{const box=$('timeline').getBoundingClientRect(),r=hitRects.filter(r=>r.info.name==='MegaMoe').reduce((a,b)=>a.duration>b.duration?a:b);$('timeline').scrollIntoView({block:'center'});const b=$('timeline').getBoundingClientRect();return {x:b.left+r.x+Math.min(20,r.w/2),y:b.top+r.y+8};})()`);
  await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...click});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...click});
  assert.ok((await evaluate(`$('inspector').textContent`)).includes('MegaMoe'));
  const overlap=await evaluate(`(()=>{const r=D.timelines[4],e=r.events.filter(e=>r.names[e[2]]==='MegaMoe').reduce((a,b)=>a[1]>b[1]?a:b),g=r.host.filter(e=>e.name==='gloo:all_reduce').reduce((a,b)=>a.duration>b.duration?a:b);return Math.max(0,Math.min(e[0]+e[1],g.start+g.duration)-Math.max(e[0],g.start))/e[1]})()`);assert.ok(overlap>.99);
  await evaluate(`$('op-search').value='HcPre';$('strict-shapes').checked=true;compare();true`);
  assert.ok((await evaluate(`$('compare-table').textContent`)).includes('没有符合条件'));
  await evaluate(`$('strict-shapes').checked=false;compare();$('preset-engram').click();true`);
  assert.ok(await evaluate('view.width<2000'));
  await evaluate(`$('preset-swa').click();window.scrollTo(0,0);true`);
  let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-desktop.png'),Buffer.from(shot.data,'base64'));
  await evaluate(`$('timeline').scrollIntoView({block:'center'});$('preset-wait').click();true`);
  shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-timeline.png'),Buffer.from(shot.data,'base64'));
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate('window.scrollTo(0,0);drawTimeline();true');
  const mobile=await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth})');assert.equal(mobile.overflow,false);
  shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(__dirname,'preview-mobile.png'),Buffer.from(shot.data,'base64'));
  assert.deepEqual(errors,[]);
  const report={passed:true,desktop,mobile,verifiedPRs:filtered,search19657:selected,waitTimelineRectangles:waiting.rectangles,megaGlooOverlapFraction:overlap,javascriptErrors:errors};
  fs.writeFileSync(path.join(__dirname,'browser_validation.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
 }finally{ws.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
