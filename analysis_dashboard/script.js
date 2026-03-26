// ─── DATA ────────────────────────────────────────────────────────────────
const ROUNDS = [1,2,3,4,5,6,7,8,9,10];
const ACC  = [0.8398,0.8549,0.8607,0.8708,0.8750,0.8708,0.8784,0.8750,0.8742,0.8800];
const F1   = [0.4824,0.5286,0.5608,0.5523,0.5895,0.5444,0.6113,0.6027,0.5989,0.6324];
const LOSS = [0.3916,0.3655,0.3550,0.3544,0.3484,0.3722,0.3738,0.3491,0.3633,0.3807];

const MISSING = [
  {col:'DEBTINC',pct:21.26,sev:'crítico'},
  {col:'DEROG',  pct:11.88,sev:'crítico'},
  {col:'DELINQ', pct:9.73, sev:'atenção'},
  {col:'MORTDUE',pct:8.69, sev:'atenção'},
  {col:'YOJ',    pct:8.64, sev:'atenção'},
  {col:'NINQ',   pct:8.56, sev:'atenção'},
  {col:'CLAGE',  pct:5.17, sev:'atenção'},
  {col:'JOB',    pct:4.68, sev:'ok'},
  {col:'REASON', pct:4.23, sev:'ok'},
  {col:'CLNO',   pct:3.72, sev:'ok'},
  {col:'VALUE',  pct:1.88, sev:'ok'},
];

const SKEW = [
  {col:'DEROG', v:5.321},{col:'DELINQ',v:4.023},{col:'NINQ',v:2.622},
  {col:'DEBTINC',v:2.852},{col:'LOAN',v:2.024},{col:'MORTDUE',v:1.814},
  {col:'VALUE',v:3.053},{col:'CLAGE',v:1.343},{col:'BAD',v:1.504},
  {col:'YOJ',v:0.988},{col:'CLNO',v:0.775},
].sort((a,b)=>b.v-a.v);

const OUTLIERS = [
  {col:'DELINQ',pct:22.32},{col:'DEROG',pct:13.80},{col:'VALUE',pct:5.47},
  {col:'LOAN',pct:4.30},{col:'MORTDUE',pct:4.30},{col:'CLNO',pct:3.82},
  {col:'NINQ',pct:3.25},{col:'DEBTINC',pct:2.00},{col:'YOJ',pct:1.67},{col:'CLAGE',pct:0.83},
];

const CORR = [
  {col:'DELINQ',r:0.3541},{col:'DEROG',r:0.2761},{col:'DEBTINC',r:0.1998},
  {col:'NINQ',r:0.1750},{col:'CLAGE',r:-0.1705},{col:'LOAN',r:-0.0751},
  {col:'YOJ',r:-0.0602},{col:'MORTDUE',r:-0.0482},{col:'VALUE',r:-0.0300},{col:'CLNO',r:-0.0042},
];

// ─── CHART DEFAULTS ──────────────────────────────────────────────────────
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size   = 11;
Chart.defaults.color       = '#888780';
Chart.defaults.plugins.legend.display = false;

const GRID = { color: 'rgba(0,0,0,0.05)', lineWidth: 0.5 };
const TICK = { color: '#888780', font: { size: 10, family: "'IBM Plex Mono', monospace" } };

function baseScales(xlabel, ylabel) {
  return {
    x: { grid: GRID, ticks: TICK, title: { display: !!xlabel, text: xlabel||'', color:'#888780', font:{size:10} } },
    y: { grid: GRID, ticks: TICK, title: { display: !!ylabel, text: ylabel||'', color:'#888780', font:{size:10} } },
  };
}

// ─── TAB SWITCHING ────────────────────────────────────────────────────────
const tabMap = [0,1,3,4,5];
function switchTab(i) {
  document.querySelectorAll('.tab-btn').forEach((b,j) => b.classList.toggle('active', i===j));
  document.querySelectorAll('.tab-panel').forEach((p,j) => p.classList.toggle('active', i===j));
  if (!chartsBuilt[i]) { buildTab(i); chartsBuilt[i] = true; }
}
const chartsBuilt = [false,false,false,false,false];

// ─── BUILD ALL TABS ON FIRST VIEW ────────────────────────────────────────
function buildTab(i) {
  if (i===0) buildTab0();
  if (i===1) buildTab1();
  if (i===2) buildTab2();
  if (i===3) buildTab3();
  if (i===4) buildTab4();
}
buildTab(0); chartsBuilt[0] = true;

// ─── TAB 0: DATASET ──────────────────────────────────────────────────────
function buildTab0() {
  new Chart(document.getElementById('c-bad'), {
    type: 'doughnut',
    data: {
      labels: ['Adimplente (BAD=0)', 'Inadimplente (BAD=1)'],
      datasets: [{
        data: [4771, 1189],
        backgroundColor: ['#1e7d55','#c0392b'],
        borderColor: '#ffffff',
        borderWidth: 2,
        hoverOffset: 6,
      }]
    },
    options: {
      cutout: '62%',
      plugins: {
        legend: { display: true, position: 'bottom', labels: { boxWidth: 12, padding: 16, font:{size:11} } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString('pt-BR')} (${(ctx.raw/5960*100).toFixed(1)}%)` } }
      }
    }
  });
}

// ─── TAB 1: MISSING ──────────────────────────────────────────────────────
function buildTab1() {
  // Bar chart
  const colors = MISSING.map(d => d.sev==='crítico'?'#c0392b': d.sev==='atenção'?'#7d5c1e':'#1e7d55');
  new Chart(document.getElementById('c-missing'), {
    type: 'bar',
    data: {
      labels: MISSING.map(d=>d.col),
      datasets: [{
        data: MISSING.map(d=>d.pct),
        backgroundColor: colors.map(c=>c+'33'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 1,
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { grid: GRID, ticks: TICK, max: 25, title:{display:true,text:'% ausente',color:'#888780',font:{size:10}} },
        y: { grid:{display:false}, ticks: TICK },
      },
      plugins: {
        annotation: { /* limiares desenhados manualmente */ },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw.toFixed(2)}% ausente` } }
      }
    },
    plugins: [{
      id: 'thresholds',
      afterDraw(chart) {
        const {ctx, chartArea: {top,bottom}, scales: {x}} = chart;
        [5, 10].forEach((v, i) => {
          const px = x.getPixelForValue(v);
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(px, top); ctx.lineTo(px, bottom);
          ctx.strokeStyle = i===0 ? '#7d5c1e' : '#c0392b';
          ctx.lineWidth = 1;
          ctx.setLineDash([4,3]);
          ctx.stroke();
          ctx.fillStyle = i===0 ? '#7d5c1e' : '#c0392b';
          ctx.font = '10px IBM Plex Mono, monospace';
          ctx.fillText(v+'%', px+3, top+12);
          ctx.restore();
        });
      }
    }]
  });

  // Progress bars
  const wrap = document.getElementById('missing-bars');
  MISSING.forEach(d => {
    const sevColor = d.sev==='crítico'?'#c0392b': d.sev==='atenção'?'#7d5c1e':'#1e7d55';
    const sevLabel = d.sev==='crítico'?'<span class="tag tag-red">crítico</span>': d.sev==='atenção'?'<span class="tag tag-amr">atenção</span>':'<span class="tag tag-grn">ok</span>';
    const row = document.createElement('div');
    row.className = 'pbar-wrap';
    row.innerHTML = `
      <span class="pbar-label">${d.col}</span>
      <div class="pbar-track"><div class="pbar-fill" style="width:0%;background:${sevColor}" data-w="${d.pct/25*100}"></div></div>
      <span class="pbar-val">${d.pct}%</span>
      <span class="pbar-sev">${sevLabel}</span>`;
    wrap.appendChild(row);
  });
  // animate bars
  setTimeout(() => {
    document.querySelectorAll('.pbar-fill[data-w]').forEach(el => {
      el.style.width = el.dataset.w + '%';
    });
  }, 100);
}

// ─── TAB 2: DISTRIBUIÇÕES ────────────────────────────────────────────────
function buildTab2() {
  // Skewness
  const skColors = SKEW.map(d => Math.abs(d.v)>1 ? '#c0392b33' : '#1e7d5533');
  const skBorders = SKEW.map(d => Math.abs(d.v)>1 ? '#c0392b' : '#1e7d55');
  new Chart(document.getElementById('c-skew'), {
    type: 'bar',
    data: {
      labels: SKEW.map(d=>d.col),
      datasets: [{ data: SKEW.map(d=>d.v), backgroundColor: skColors, borderColor: skBorders, borderWidth: 1.5, borderRadius: 1 }]
    },
    options: {
      scales: {
        ...baseScales('Variável','Assimetria (skew)'),
        y: { grid: GRID, ticks: TICK }
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` skew = ${ctx.raw.toFixed(3)}` } } }
    },
    plugins: [{
      id: 'threshold',
      afterDraw(chart) {
        const {ctx, chartArea:{left,right}, scales:{y}} = chart;
        [1,-1].forEach(v => {
          const py = y.getPixelForValue(v);
          ctx.save();
          ctx.beginPath(); ctx.moveTo(left,py); ctx.lineTo(right,py);
          ctx.strokeStyle='#c0392b'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
          ctx.stroke();
          ctx.fillStyle='#c0392b'; ctx.font='9px IBM Plex Mono,monospace';
          ctx.fillText(v===1?'limiar +1':'limiar −1', left+4, py-3);
          ctx.restore();
        });
      }
    }]
  });

  // Outliers
  const outColors = OUTLIERS.map(d => d.pct>10?'#c0392b33':d.pct>5?'#7d5c1e33':'#1a3a6b22');
  const outBorders = OUTLIERS.map(d => d.pct>10?'#c0392b':d.pct>5?'#7d5c1e':'#1a3a6b');
  new Chart(document.getElementById('c-outliers'), {
    type: 'bar',
    data: {
      labels: OUTLIERS.map(d=>d.col),
      datasets: [{ data: OUTLIERS.map(d=>d.pct), backgroundColor: outColors, borderColor: outBorders, borderWidth: 1.5, borderRadius: 1 }]
    },
    options: {
      scales: {
        ...baseScales('Variável','% outliers (IQR×1.5)'),
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw.toFixed(2)}% outliers` } } }
    }
  });

  // REASON
  new Chart(document.getElementById('c-reason'), {
    type: 'bar',
    data: {
      labels: ['DebtCon','HomeImp','NaN'],
      datasets: [{ data: [65.91, 29.87, 4.23], backgroundColor: ['#1a3a6b33','#1e7d5533','#88878033'], borderColor: ['#1a3a6b','#1e7d55','#888780'], borderWidth: 1.5, borderRadius: 1 }]
    },
    options: {
      scales: { ...baseScales('','%') },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } } }
    }
  });

  // JOB
  new Chart(document.getElementById('c-job'), {
    type: 'bar',
    data: {
      labels: ['Other','ProfExe','Office','Mgr','NaN','Self','Sales'],
      datasets: [{ data: [40.07,21.41,15.91,12.87,4.68,3.24,1.83], backgroundColor: '#1a3a6b22', borderColor: '#1a3a6b', borderWidth: 1.5, borderRadius: 1 }]
    },
    options: {
      scales: { ...baseScales('','%') },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } } }
    }
  });
}

// ─── TAB 3: CORRELAÇÕES ──────────────────────────────────────────────────
function buildTab3() {
  const corrColors = CORR.map(d => d.r > 0 ? '#c0392b' : '#1a3a6b');
  const corrBg     = CORR.map(d => d.r > 0 ? '#c0392b22' : '#1a3a6b22');
  new Chart(document.getElementById('c-corr'), {
    type: 'bar',
    data: {
      labels: CORR.map(d=>d.col),
      datasets: [{
        data: CORR.map(d=>d.r),
        backgroundColor: corrBg,
        borderColor: corrColors,
        borderWidth: 1.5,
        borderRadius: 1,
      }]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: { grid:GRID, ticks:TICK, min:-0.4, max:0.4, title:{display:true,text:'Correlação de Pearson (r)',color:'#888780',font:{size:10}} },
        y: { grid:{display:false}, ticks:TICK },
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` r = ${ctx.raw.toFixed(4)}` } } }
    },
    plugins: [{
      id:'zeroline',
      afterDraw(chart){
        const {ctx,chartArea:{top,bottom},scales:{x}}=chart;
        const px=x.getPixelForValue(0);
        ctx.save();ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,bottom);
        ctx.strokeStyle='#888780';ctx.lineWidth=1;ctx.stroke();ctx.restore();
      }
    }]
  });
}

// ─── TAB 4: FEDERADO ─────────────────────────────────────────────────────
function buildTab4() {
  // Accuracy + F1
  new Chart(document.getElementById('c-fed-acc'), {
    type: 'line',
    data: {
      labels: ROUNDS,
      datasets: [
        { label:'Acurácia', data: ACC.map(v=>+(v*100).toFixed(2)), borderColor:'#1a3a6b', backgroundColor:'#1a3a6b18', tension:0.35, fill:true, pointRadius:4, pointHoverRadius:6, borderWidth:1.5 },
        { label:'F1-Score', data: F1.map(v=>+(v*100).toFixed(2)), borderColor:'#1e7d55', backgroundColor:'#1e7d5518', tension:0.35, fill:true, pointRadius:4, pointHoverRadius:6, borderWidth:1.5 },
      ]
    },
    options: {
      scales: {
        ...baseScales('Round (comunicação)','%'),
        y: { grid:GRID, ticks:{...TICK, callback:v=>v+'%'}, min:40, max:95 }
      },
      plugins: {
        legend:{ display:true, position:'bottom', labels:{boxWidth:12,padding:16,font:{size:10}} },
        tooltip:{ callbacks:{label:ctx=>` ${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`} }
      },
      interaction:{ mode:'index', intersect:false }
    }
  });

  // Loss
  new Chart(document.getElementById('c-fed-loss'), {
    type: 'line',
    data: {
      labels: ROUNDS,
      datasets: [{
        label:'Loss distribuída',
        data: LOSS,
        borderColor:'#c0392b',
        backgroundColor:'#c0392b18',
        tension:0.35,
        fill:true,
        pointRadius:4,
        pointHoverRadius:6,
        borderWidth:1.5,
      }]
    },
    options: {
      scales: {
        ...baseScales('Round','BCE Loss'),
        y: { grid:GRID, ticks:{...TICK, callback:v=>v.toFixed(2)}, min:0.30, max:0.42 }
      },
      plugins: { tooltip:{ callbacks:{ label:ctx=>` loss = ${ctx.raw.toFixed(4)}` } } }
    }
  });

  // Table
  const tbody = document.getElementById('fed-table');
  ACC.forEach((acc,i) => {
    const delta = i===0 ? '—' : (((acc - ACC[i-1])*100).toFixed(2)+'pp');
    const dColor = i===0?'var(--muted)': acc>ACC[i-1]?'var(--accent3)':'var(--accent2)';
    tbody.innerHTML += `<tr>
      <td>${i+1}</td>
      <td>${(acc*100).toFixed(2)}%</td>
      <td>${F1[i].toFixed(4)}</td>
      <td>${LOSS[i].toFixed(4)}</td>
      <td style="color:${dColor}">${delta}</td>
    </tr>`;
  });

  // Scatter acc x f1
  new Chart(document.getElementById('c-fed-scatter'), {
    type: 'scatter',
    data: {
      datasets: [{
        data: ACC.map((a,i)=>({x:+(a*100).toFixed(2), y:+F1[i].toFixed(4), r:i+1})),
        backgroundColor: ROUNDS.map((_,i)=>`rgba(26,58,107,${0.3+i*0.07})`),
        borderColor: '#1a3a6b',
        borderWidth: 1,
        pointRadius: 7,
        pointHoverRadius: 9,
      }]
    },
    options: {
      scales: {
        x:{ grid:GRID, ticks:{...TICK,callback:v=>v+'%'}, min:83, max:89.5, title:{display:true,text:'Acurácia (%)',color:'#888780',font:{size:10}} },
        y:{ grid:GRID, ticks:TICK, min:0.45, max:0.65, title:{display:true,text:'F1-Score',color:'#888780',font:{size:10}} },
      },
      plugins:{
        tooltip:{callbacks:{label:ctx=>`Round ${ctx.raw.r}: acc=${ctx.raw.x}% · F1=${ctx.raw.y}`}},
      }
    },
    plugins:[{
      id:'labels',
      afterDatasetsDraw(chart){
        const {ctx,data}=chart;
        data.datasets[0].data.forEach((pt,i)=>{
          const meta=chart.getDatasetMeta(0);
          const {x,y}=meta.data[i].getProps(['x','y'],true);
          ctx.save();ctx.fillStyle='#1a3a6b';ctx.font='9px IBM Plex Mono,monospace';
          ctx.textAlign='center';ctx.fillText('R'+pt.r,x,y-10);ctx.restore();
        });
      }
    }]
  });
}