/* PROJETO - Visualizador de Rede Neural (HMEQ Credit Risk)
* =========================================================
* Descrição: Interface interativa em JavaScript utilizando Three.js
* para visualização de uma rede neural feedforward.
*
* A aplicação simula um forward pass em tempo real, exibindo:
* - Ativações dos neurônios
* - Intensidade das conexões
* - Probabilidade de inadimplência
*
* Inclui interação 3D (rotação/zoom), geração de inputs aleatórios
* e modos de execução (idle, manual, automático).
*
* Objetivo: Demonstrar visualmente o funcionamento interno de uma
* rede neural, focando em aprendizado e apresentação.
* ========================================================= */


/* ──────────────────────────────────────────────────────────
   DADOS DA REDE (FEATURES + ARQUITETURA)
────────────────────────────────────────── */
const FEATURES = [
  {name:'LOAN',   desc:'Valor do empréstimo', color:'#378ADD'},
  {name:'MORTDUE',desc:'Dívida hipoteca',     color:'#378ADD'},
  {name:'VALUE',  desc:'Valor do imóvel',     color:'#378ADD'},
  {name:'REASON', desc:'Motivo (0/1)',         color:'#7F77DD'},
  {name:'JOB',    desc:'Categoria profiss.',  color:'#7F77DD'},
  {name:'YOJ',    desc:'Anos no emprego',     color:'#378ADD'},
  {name:'DEROG',  desc:'Reg. depreciativos',  color:'#378ADD'},
  {name:'DELINQ', desc:'Créditos em atraso',  color:'#378ADD'},
  {name:'CLAGE',  desc:'Idade linha crédito', color:'#378ADD'},
  {name:'NINQ',   desc:'Consultas recentes',  color:'#378ADD'},
  {name:'CLNO',   desc:'Nº linhas crédito',   color:'#378ADD'},
  {name:'DEBTINC',desc:'Dívida/renda',        color:'#378ADD'},
];

// Estrutura da rede neural
const LAYER_SIZES = [12, 16, 8, 1];
const LAYER_COLORS = [0x378ADD, 0x7F77DD, 0x1D9E75, 0xD85A30];
const LAYER_NAMES = ['Entrada','Oculta 1','Oculta 2','Saída'];

/* ──────────────────────────────────────────────────────────
   CONFIGURAÇÃO DA CENA (THREE.JS)
────────────────────────────────────────── */
const container = document.getElementById('canvas-container');
const W = window.innerWidth, H = window.innerHeight;
// Renderer principal
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.setClearColor(0x080c14, 1);
container.appendChild(renderer.domElement);
// Cena + câmera
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 1000);
camera.position.set(0, 0, 22);

/* ──────────────────────────────────────────────────────────
   ILUMINAÇÃO
────────────────────────────────────────── */
// Luz ambiente (base)
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
// Luz direcional principal
const dLight = new THREE.DirectionalLight(0x4488ff, 0.8);
dLight.position.set(5, 10, 10);
scene.add(dLight);
// Luz secundária para contraste
const dLight2 = new THREE.DirectionalLight(0xff8844, 0.3);
dLight2.position.set(-5, -5, 5);
scene.add(dLight2);

/* ──────────────────────────────────────────────────────────
   CONSTRUÇÃO DA REDE (NÓS + ARESTAS)
────────────────────────────────────────── */
const layerXPositions = [-9, -3, 3, 9];
// Estruturas principais
const nodeObjects = [];   // nodeObjects[layer][node] = mesh
const edgeObjects = [];   // edgeObjects[layer] = [line, ...]
const activations = LAYER_SIZES.map(n => new Array(n).fill(0));
const nodePositions = [];

// Geometria padrão dos neurônios
const sphereGeo = new THREE.SphereGeometry(0.32, 16, 16);
/* FUNÇÃO: CRIAR MATERIAL DO NÓ */
function makeNodeMat(hex, active) {
  return new THREE.MeshPhongMaterial({
    color: hex,
    emissive: hex,
    emissiveIntensity: active ? 0.6 : 0.05,
    transparent: true,
    opacity: active ? 1.0 : 0.35,
    shininess: 80,
  });
}

/* ──────────────────────────────────────────────────────────
   CRIAÇÃO DOS NÓS (NEURÔNIOS)
────────────────────────────────────────── */
for (let l = 0; l < LAYER_SIZES.length; l++) {
  const n = LAYER_SIZES[l];
  const col = LAYER_COLORS[l];
  nodeObjects.push([]);
  nodePositions.push([]);
  const maxShow = Math.min(n, n); // show all
  const totalH = (maxShow - 1) * 1.15;

  for (let i = 0; i < maxShow; i++) {
    const y = totalH/2 - i * 1.15;
    const mat = makeNodeMat(col, false);
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(layerXPositions[l], y, 0);
    mesh.userData = { layer: l, node: i, baseColor: col };
    scene.add(mesh);
    nodeObjects[l].push(mesh);
    nodePositions[l].push(new THREE.Vector3(layerXPositions[l], y, 0));
  }
}

/* ──────────────────────────────────────────────────────────
   CRIAÇÃO DAS CONEXÕES (ARESTAS)
────────────────────────────────────────── */
const edgeMat = new THREE.LineBasicMaterial({
  color: 0x2255aa,
  transparent: true,
  opacity: 0.07,
  linewidth: 1,
});

const allEdges = [];

for (let l = 0; l < LAYER_SIZES.length - 1; l++) {
  edgeObjects.push([]);
  const fromCount = Math.min(LAYER_SIZES[l], LAYER_SIZES[l]);
  const toCount   = Math.min(LAYER_SIZES[l+1], LAYER_SIZES[l+1]);

  for (let i = 0; i < fromCount; i++) {
    for (let j = 0; j < toCount; j++) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        nodePositions[l][i],
        nodePositions[l+1][j],
      ]);
      const mat = new THREE.LineBasicMaterial({
        color: 0x3b8fff,
        transparent: true,
        opacity: 0.05,
      });
      const line = new THREE.Line(geo, mat);
      scene.add(line);
      edgeObjects[l].push({ line, from: i, to: j, mat });
      allEdges.push({ line, from_l: l, from_i: i, to_l: l+1, to_i: j, mat });
    }
  }
}

// Layer label sprites (textura do canvas)
function makeLabel(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 200; canvas.height = 40;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 200, 40);
  ctx.fillStyle = color;
  ctx.font = '500 14px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, 100, 26);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(3, 0.6, 1);
  return sprite;
}

const labelColors = ['#378ADD','#7F77DD','#1D9E75','#D85A30'];
LAYER_NAMES.forEach((name, l) => {
  const maxY = nodePositions[l].reduce((m, p) => Math.max(m, p.y), -Infinity);
  const sp = makeLabel(name + ' (' + LAYER_SIZES[l] + ')', labelColors[l]);
  sp.position.set(layerXPositions[l], maxY + 1.6, 0);
  scene.add(sp);
});

/* ──────────────────────────────────────────────────────────
   INTERAÇÃO (MOUSE / TOUCH / ZOOM)
────────────────────────────────────────── */
let isDragging = false, prevMouse = {x:0, y:0};
let spherical = {theta: 0, phi: 0};
const networkGroup = new THREE.Group();
scene.children.filter(c => c !== dLight && c !== dLight2).forEach(c => networkGroup.add(c));
// Reconstruir
scene.clear();
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
scene.add(dLight); scene.add(dLight2);

const pivot = new THREE.Group();
scene.add(pivot);

// Re-adiciona todos os objetos para pivot
nodeObjects.flat().forEach(m => pivot.add(m));
allEdges.forEach(e => pivot.add(e.line));
// Re-add labels
const labelSprites = [];
LAYER_NAMES.forEach((name, l) => {
  const maxY = nodePositions[l].reduce((m, p) => Math.max(m, p.y), -Infinity);
  const sp = makeLabel(name + ' (' + LAYER_SIZES[l] + ')', labelColors[l]);
  sp.position.set(layerXPositions[l], maxY + 1.6, 0);
  pivot.add(sp);
  labelSprites.push(sp);
});

renderer.domElement.addEventListener('mousedown', e => {
  isDragging = true;
  prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const dx = e.clientX - prevMouse.x;
  const dy = e.clientY - prevMouse.y;
  spherical.theta -= dx * 0.008;
  spherical.phi   -= dy * 0.005;
  spherical.phi = Math.max(-0.8, Math.min(0.8, spherical.phi));
  prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('mouseup', () => isDragging = false);
renderer.domElement.addEventListener('wheel', e => {
  camera.position.z = Math.max(10, Math.min(40, camera.position.z + e.deltaY * 0.03));
});

// Touch support
renderer.domElement.addEventListener('touchstart', e => {
  isDragging = true;
  prevMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
});
renderer.domElement.addEventListener('touchmove', e => {
  if (!isDragging) return;
  const dx = e.touches[0].clientX - prevMouse.x;
  const dy = e.touches[0].clientY - prevMouse.y;
  spherical.theta -= dx * 0.008;
  spherical.phi   -= dy * 0.005;
  spherical.phi = Math.max(-0.8, Math.min(0.8, spherical.phi));
  prevMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
});
renderer.domElement.addEventListener('touchend', () => isDragging = false);

/* ──────────────────────────────────────────────────────────
   INPUTS (FEATURES)
────────────────────────────────────────── */
let inputValues = new Array(12).fill(0.5);

function buildFeaturesPanel() {
  const el = document.getElementById('features-list');
  el.innerHTML = '';
  FEATURES.forEach((f, i) => {
    const row = document.createElement('div');
    row.className = 'input-feature';
    row.title = f.desc;
    row.innerHTML = `
      <span class="feat-name">${f.name}</span>
      <div class="feat-bar-wrap"><div class="feat-bar" id="fbar-${i}" style="width:${inputValues[i]*100}%"></div></div>
      <span class="feat-val" id="fval-${i}">${inputValues[i].toFixed(2)}</span>`;
    el.appendChild(row);
  });
}
buildFeaturesPanel();

function updateFeaturesPanel() {
  FEATURES.forEach((_, i) => {
    const bar = document.getElementById('fbar-'+i);
    const val = document.getElementById('fval-'+i);
    if (bar) bar.style.width = (inputValues[i]*100)+'%';
    if (val) val.textContent = inputValues[i].toFixed(2);
  });
}

function randomInput() {
  inputValues = Array.from({length:12}, () => Math.random());
  updateFeaturesPanel();
  if (currentMode === 'auto' || currentMode === 'idle') {
    runForwardPass();
  }
}

/* ──────────────────────────────────────────────────────────
   FORWARD PASS (SIMULAÇÃO)
────────────────────────────────────────── */
let currentMode = 'idle';
let animating = false;
let autoInterval = null;

function setMode(m) {
  currentMode = m;
  ['idle','forward','auto'].forEach(id => {
    document.getElementById('btn-'+id).classList.toggle('active', id === m);
  });
  if (autoInterval) { clearInterval(autoInterval); autoInterval = null; }

  if (m === 'forward') { runForwardPass(); }
  if (m === 'auto') {
    runForwardPass();
    autoInterval = setInterval(() => {
      randomInput();
      runForwardPass();
    }, 2800);
  }
  if (m === 'idle') { resetNetwork(); }
}

function resetNetwork() {
  nodeObjects.flat().forEach(mesh => {
    mesh.material.emissiveIntensity = 0.05;
    mesh.material.opacity = 0.35;
  });
  allEdges.forEach(e => {
    e.mat.opacity = 0.05;
    e.mat.color.setHex(0x3b8fff);
  });
  document.getElementById('out-prob').textContent = '—';
  document.getElementById('out-verdict').textContent = 'aguardando sinal';
  document.getElementById('out-verdict').style.color = 'var(--muted)';
}

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function relu(x) { return Math.max(0, x); }

async function runForwardPass() {
  if (animating) return;
  animating = true;

  // Ativação Simulada
  let acts = [inputValues.slice()];
  for (let l = 0; l < LAYER_SIZES.length - 1; l++) {
    const prev = acts[l];
    const next = [];
    const nextSize = LAYER_SIZES[l+1];
    for (let j = 0; j < nextSize; j++) {
      let sum = prev.reduce((s, v, i) => s + v * (Math.random() * 0.8 + 0.1), 0) / prev.length;
      next.push(l === LAYER_SIZES.length - 2 ? sigmoid(sum * 2 - 1) : relu(sum));
    }
    acts.push(next);
  }

  const finalProb = acts[acts.length-1][0];

  // Animação camada por camada
  for (let l = 0; l < LAYER_SIZES.length; l++) {
    // Ilumine as bordas nesta camada
    nodeObjects[l].forEach((mesh, i) => {
      const a = Math.min(1, acts[l][i]);
      mesh.material.emissiveIntensity = 0.1 + a * 0.9;
      mesh.material.opacity = 0.4 + a * 0.6;
    });

    // Ilumine as bordas a partir desta camada.
    if (l < LAYER_SIZES.length - 1) {
      allEdges.filter(e => e.from_l === l).forEach(e => {
        const w = acts[l][e.from_i] * acts[l+1][e.to_i];
        e.mat.opacity = 0.03 + w * 0.55;
        e.mat.color.setHex(w > 0.4 ? 0x00e5a0 : 0x3b8fff);
      });
    }

    await sleep(l === 0 ? 100 : 250);
  }

  // Update saida do painel
  const pct = (finalProb * 100).toFixed(1);
  document.getElementById('out-prob').textContent = pct + '%';
  const verdict = finalProb > 0.5
    ? { text: '⚠ Inadimplente', color: '#ff6b6b' }
    : { text: '✓ Adimplente',   color: '#00e5a0' };
  document.getElementById('out-verdict').textContent = verdict.text;
  document.getElementById('out-verdict').style.color = verdict.color;

  await sleep(800);

  // Fade back out
  nodeObjects.flat().forEach(mesh => {
    mesh.material.emissiveIntensity = 0.05;
    mesh.material.opacity = 0.35;
  });
  allEdges.forEach(e => {
    e.mat.opacity = 0.05;
    e.mat.color.setHex(0x3b8fff);
  });

  animating = false;
  if (currentMode === 'forward') {
    document.getElementById('btn-forward').classList.remove('active');
    currentMode = 'idle';
    document.getElementById('btn-idle').classList.add('active');
  }
}

/* ──────────────────────────────────────────────────────────
   LOOP DE RENDERIZAÇÃO
────────────────────────────────────────── */
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  // Auto-rotate quando idle
  if (!isDragging) {
    if (currentMode === 'idle') spherical.theta += 0.003;
  }

  // Aplicar rotação esferica para pivot
  pivot.rotation.y = spherical.theta;
  pivot.rotation.x = spherical.phi;

  // Pulso sutil em todos os nodes
  nodeObjects.flat().forEach((mesh, idx) => {
    if (mesh.material.opacity < 0.4) {
      const pulse = 0.02 * Math.sin(t * 1.5 + idx * 0.3);
      mesh.material.emissiveIntensity = 0.05 + pulse;
    }
  });

  renderer.render(scene, camera);
}
animate();

/* ──────────────────────────────────────────────────────────
   RESPONSIVIDADE
────────────────────────────────────────── */
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});