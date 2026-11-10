(() => {
 'use strict';
 const byId = id => document.getElementById(id);
 const data = JSON.parse(byId('learning-data').textContent);
 const progressKey = 'npu-inference-learning-v1';
 let saved = [];
 try { saved = JSON.parse(localStorage.getItem(progressKey) || '[]'); } catch (_) { saved = []; }
 if (!Array.isArray(saved)) saved = [];
 const progressInputs = [...document.querySelectorAll('.phase input')];
 function updateProgress() {
   const state = progressInputs.map(input => input.checked);
   const count = state.filter(Boolean).length;
   byId('progress-count').textContent = `${count} / ${state.length} 阶段完成`;
   byId('progress-fill').style.width = `${100 * count / state.length}%`;
   try { localStorage.setItem(progressKey, JSON.stringify(state)); } catch (_) { /* File reading still works. */ }
 }
 progressInputs.forEach((input, index) => {
   input.checked = Boolean(saved[index]);
   input.addEventListener('change', updateProgress);
 });
 byId('reset-progress').addEventListener('click', () => {
   progressInputs.forEach(input => { input.checked = false; });
   updateProgress();
 });
 updateProgress();

 const mapButtons = [...document.querySelectorAll('.map-button')];
 function selectModule(index) {
   const module = data.modules[index];
   mapButtons.forEach((button, i) => button.setAttribute('aria-selected', String(i === index)));
   byId('module-title').textContent = module.question;
   byId('module-topics').replaceChildren(...module.topics.map(topic => {
     const item = document.createElement('li'); item.textContent = topic; return item;
   }));
   byId('module-code').textContent = module.code;
   byId('module-experiment').textContent = module.experiment;
   byId('module-link').href = '#' + module.anchor;
 }
 mapButtons.forEach((button, index) => {
   button.addEventListener('click', () => selectModule(index));
   button.addEventListener('keydown', event => {
     if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
     event.preventDefault();
     const next = (index + (event.key === 'ArrowRight' ? 1 : -1) + mapButtons.length) % mapButtons.length;
     selectModule(next); mapButtons[next].focus();
   });
 });
 selectModule(0);

 function showSymptom() {
   const item = data.symptoms[byId('symptom').value];
   byId('triage-title').textContent = item.title;
   byId('triage-checks').replaceChildren(...item.checks.map(check => {
     const li = document.createElement('li'); li.textContent = check; return li;
   }));
   byId('triage-next').textContent = item.next;
 }
 byId('symptom').addEventListener('change', showSymptom);
 showSymptom();

 function drawRoofline() {
   const compute = Number(byId('roof-compute').value);
   const bandwidth = Number(byId('roof-bandwidth').value);
   const intensity = Number(byId('roof-intensity').value);
   const observed = Number(byId('roof-observed').value);
   const values = [compute, bandwidth, intensity, observed];
   if (values.some(value => !Number.isFinite(value) || value <= 0)) {
     byId('roof-summary').textContent = '请输入有限的正数。';
     byId('roof-summary').classList.add('roof-error');
     return;
   }
   const ceiling = Math.min(compute, bandwidth * intensity);
   const memorySide = bandwidth * intensity < compute;
   const ratio = observed / ceiling;
   const lead = memorySide ? '模型在此算术强度上的上界来自带宽侧' : '模型在此算术强度上的上界来自算力侧';
   byId('roof-summary').textContent = `${lead}：${ceiling.toFixed(2)} TFLOP/s。输入性能为该上界的 ${(100 * ratio).toFixed(1)}%。` +
     (ratio > 1.001 ? ' 输入点超出这个简化上界，请核对流量、运算量、单位、设备范围及上限假设。' : ' 这不是瓶颈诊断；还需要实测流量、流水计数器和关键路径证据。');
   byId('roof-summary').classList.toggle('roof-error', ratio > 1.001);
   const width = 640, height = 328, left = 66, right = 18, top = 20, bottom = 54;
   const maxX = Math.max(4, Math.ceil(Math.log10(intensity)) + 1);
   const minX = Math.min(-2, Math.floor(Math.log10(intensity)) - 1);
   const maxY = Math.max(2, Math.ceil(Math.log10(Math.max(compute, observed))) + .2);
   const minY = Math.min(-2, Math.floor(Math.log10(Math.min(ceiling, observed))) - .3);
   const x = value => left + (Math.log10(value) - minX) / (maxX - minX) * (width - left - right);
   const y = value => height - bottom - (Math.log10(value) - minY) / (maxY - minY) * (height - bottom - top);
   const elements = [];
   for (let power = Math.ceil(minX); power <= Math.floor(maxX); power++) {
     const value = 10 ** power, px = x(value);
     elements.push(`<line x1="${px}" y1="${top}" x2="${px}" y2="${height-bottom}" stroke="#dfe8df"/><text x="${px}" y="${height-bottom+20}" text-anchor="middle" font-size="11" fill="#52716d">${value}</text>`);
   }
   for (let power = Math.ceil(minY); power <= Math.floor(maxY); power++) {
     const value = 10 ** power, py = y(value);
     elements.push(`<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}" stroke="#dfe8df"/><text x="${left-10}" y="${py+4}" text-anchor="end" font-size="11" fill="#52716d">${value}</text>`);
   }
   const points = [];
   for (let i = 0; i <= 160; i++) {
     const value = 10 ** (minX + (maxX - minX) * i / 160);
     const performance = Math.min(compute, bandwidth * value);
     if (performance >= 10 ** minY) points.push(`${x(value)},${y(performance)}`);
   }
   elements.push(`<polyline points="${points.join(' ')}" fill="none" stroke="#087f82" stroke-width="3"/>`);
   elements.push(`<line x1="${x(intensity)}" y1="${y(ceiling)}" x2="${x(intensity)}" y2="${y(observed)}" stroke="#ba7c32" stroke-dasharray="4 4"/>`);
   elements.push(`<circle cx="${x(intensity)}" cy="${y(ceiling)}" r="5" fill="#087f82"/><circle cx="${x(intensity)}" cy="${y(observed)}" r="6" fill="#c77b28"/>`);
   elements.push(`<text x="${left+9}" y="${top+13}" font-size="12" fill="#087f82">绿色：假设上界　橙色：输入的性能</text>`);
   elements.push(`<text x="${width/2}" y="${height-9}" text-anchor="middle" font-size="12" fill="#294d4b">算术强度 FLOPs / byte（对数轴）</text>`);
   elements.push(`<text transform="translate(17 ${height/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#294d4b">性能 TFLOP/s（对数轴）</text>`);
   byId('roof-svg').innerHTML = elements.join('');
 }
 ['compute','bandwidth','intensity','observed'].forEach(name => byId('roof-' + name).addEventListener('input', drawRoofline));
 drawRoofline();

 if ('IntersectionObserver' in window) {
   const links = [...document.querySelectorAll('.rail nav a')];
   const observer = new IntersectionObserver(entries => {
     const visible = entries.filter(entry => entry.isIntersecting);
     if (!visible.length) return;
     const id = visible[0].target.id;
     links.forEach(link => link.classList.toggle('active', link.hash === '#' + id));
   }, { rootMargin: '-5% 0px -65% 0px' });
   document.querySelectorAll('article>h2[id], section[id]').forEach(section => observer.observe(section));
 }
 window.learningGuideReady = true;
})();
