/* app.js — Face Sentinel Web Frontend */
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
var activeCams  = [];     // array of camera indices currently open
var allFaces    = [];     // current user's face records
var faceSort    = 'new';

var CAM_COLORS  = ['#00ff41','#50ffc8','#00dcff','#6464ff','#ffc800','#c8ff64'];
var CAM_COLORS_CSS = [
  'rgba(0,255,65,.6)','rgba(80,255,200,.6)','rgba(0,220,255,.6)',
  'rgba(100,100,255,.6)','rgba(255,200,0,.6)','rgba(200,255,100,.6)'
];

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  loadActiveCams();
  startSSE();
  if (window.FS_CONFIG && window.FS_CONFIG.is_master) {
    loadGitHubConfig();
  }
});

// ── Tab switching ─────────────────────────────────────────────────────────────
window.switchTab = function (name) {
  document.querySelectorAll('.pane').forEach(function(p){ p.classList.remove('active'); });
  document.querySelectorAll('.tab').forEach(function(t){ t.classList.remove('active'); });
  var pane = document.getElementById('pane-' + name);
  var tab  = document.getElementById('tab-'  + name);
  if (pane) pane.classList.add('active');
  if (tab)  tab.classList.add('active');

  if (name === 'reports') loadReports();
  if (name === 'users')   loadUsers();
};

// ── Logout ────────────────────────────────────────────────────────────────────
window.logout = async function () {
  await fetch('/api/auth/logout', {method:'POST'});
  window.location.href = '/login';
};

// ── Camera scan & open ────────────────────────────────────────────────────────
window.scanCameras = async function () {
  var btn = document.getElementById('btn-scan');
  btn.textContent = '⟳ Escaneando…';
  btn.disabled    = true;
  var list = document.getElementById('cam-checklist');
  list.innerHTML  = '<div style="color:var(--gray);font-size:11px;padding:4px 0">Aguarde…</div>';

  try {
    var r    = await fetch('/api/cameras/scan');
    var cams = await r.json();

    if (!cams.length) {
      list.innerHTML = '<div style="color:var(--red);font-size:11px;padding:4px 0">Nenhuma câmera encontrada.</div>';
    } else {
      list.innerHTML = '';
      cams.forEach(function (cam, i) {
        var row   = document.createElement('div');
        row.className = 'cam-check-row';
        var color = CAM_COLORS[i % CAM_COLORS.length];
        row.style.borderColor = color + '66';
        row.innerHTML =
          '<input type="checkbox" id="cc-'+cam.index+'" value="'+cam.index+'" checked>' +
          '<label for="cc-'+cam.index+'" style="color:'+color+'">' +
            cam.label + ' &nbsp;(' + cam.fps.toFixed(0) + ' fps)</label>';
        list.appendChild(row);
      });
      document.getElementById('btn-open-cams').style.display = 'block';
    }
  } catch(e) {
    list.innerHTML = '<div style="color:var(--red);font-size:11px">Erro: ' + e.message + '</div>';
  } finally {
    btn.textContent = '⊕ Escanear câmeras';
    btn.disabled    = false;
  }
};

window.openSelected = async function () {
  var checks  = document.querySelectorAll('#cam-checklist input[type=checkbox]:checked');
  var indices = Array.from(checks).map(function(c){ return parseInt(c.value); });
  if (!indices.length) { toast('Selecione pelo menos uma câmera.', true); return; }

  var r    = await fetch('/api/cameras/open', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({indices: indices})
  });
  var data = await r.json();

  data.opened.forEach(function(idx) {
    if (!activeCams.includes(idx)) activeCams.push(idx);
  });
  renderVideoGrid();
  renderActiveCamsList();
  document.getElementById('cam-checklist').innerHTML = '';
  document.getElementById('btn-open-cams').style.display = 'none';
  toast('Câmera(s) abertas: ' + data.opened.join(', '));
};

window.closeCamera = async function (idx) {
  await fetch('/api/cameras/close', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: idx})
  });
  activeCams = activeCams.filter(function(i){ return i !== idx; });
  renderVideoGrid();
  renderActiveCamsList();
};

async function loadActiveCams() {
  try {
    var r    = await fetch('/api/cameras/active');
    var cams = await r.json();
    activeCams = cams.map(function(c){ return c.index; });
    renderVideoGrid();
    renderActiveCamsList();
  } catch(e) {}
}

// ── Video grid rendering ──────────────────────────────────────────────────────
function renderVideoGrid() {
  var grid = document.getElementById('video-grid');
  grid.innerHTML = '';

  if (!activeCams.length) {
    var classes = ['g1'];
    grid.className = 'video-grid g1';
    grid.innerHTML =
      '<div class="no-signal">' +
        '<span>◈ SEM SINAL ◈</span>' +
        '<small>Clique em ⊕ Escanear câmeras para começar</small>' +
      '</div>';
    return;
  }

  var n = activeCams.length;
  var cls = n === 1 ? 'g1' : n === 2 ? 'g2' : n <= 4 ? 'g4' : 'g3';
  grid.className = 'video-grid ' + cls;

  activeCams.forEach(function (idx, slot) {
    var color = CAM_COLORS[slot % CAM_COLORS.length];
    var cell  = document.createElement('div');
    cell.className   = 'video-cell';
    cell.id          = 'cell-' + idx;
    cell.style.borderColor = color + '99';

    var img = document.createElement('img');
    img.id  = 'feed-' + idx;
    img.src = '/video_feed/' + idx + '?mode=face&conf=50&scale=115&neigh=5&scanlines=1';
    img.alt = 'Camera ' + idx;

    var lbl = document.createElement('div');
    lbl.className   = 'cam-label';
    lbl.style.color = color;
    lbl.textContent = 'CAM ' + idx;

    cell.appendChild(img);
    cell.appendChild(lbl);
    grid.appendChild(cell);
  });
}

function renderActiveCamsList() {
  var el = document.getElementById('active-cams');
  el.innerHTML = '';
  if (!activeCams.length) {
    el.innerHTML = '<div style="color:var(--gray);font-size:11px;padding:4px 0">Nenhuma câmera ativa</div>';
    return;
  }
  activeCams.forEach(function(idx, slot) {
    var color = CAM_COLORS[slot % CAM_COLORS.length];
    var row   = document.createElement('div');
    row.className     = 'active-cam-row';
    row.style.borderColor = color + '66';
    row.innerHTML =
      '<span class="active-cam-label" style="color:'+color+'">● Camera ' + idx + '</span>' +
      '<button class="close-cam-btn" onclick="closeCamera('+idx+')">✕</button>';
    el.appendChild(row);
  });
}

// ── Detection settings ────────────────────────────────────────────────────────
var _settingsTimer = null;
window.applySettings = function () {
  clearTimeout(_settingsTimer);
  _settingsTimer = setTimeout(function () {
    var mode  = document.querySelector('input[name=mode]:checked')?.value || 'face';
    var conf  = parseInt(document.getElementById('sl-conf').value)  / 100;
    var scale = parseInt(document.getElementById('sl-scale').value) / 100;
    var neigh = parseInt(document.getElementById('sl-neigh').value);
    var heatmap  = document.getElementById('opt-heatmap').checked;
    var scanlines = document.getElementById('opt-scanlines').checked;

    fetch('/api/detection/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mode, confidence:conf, scale, min_neighbors:neigh, show_heatmap:heatmap, scanlines})
    });

    // update MJPEG src with new params for each active feed
    activeCams.forEach(function(idx) {
      var img = document.getElementById('feed-' + idx);
      if (!img) return;
      var base = '/video_feed/' + idx;
      img.src  = base + '?mode='+mode+'&conf='+(conf*100|0)+
                        '&scale='+(scale*100|0)+'&neigh='+neigh+
                        '&heatmap='+(heatmap?1:0)+'&scanlines='+(scanlines?1:0)+'&t='+Date.now();
    });
  }, 300);
};

// ── SSE live stats ────────────────────────────────────────────────────────────
function startSSE() {
  var es = new EventSource('/api/stream/stats');
  es.onmessage = function (e) {
    try {
      var data = JSON.parse(e.data);
      renderStats(data);
      var any = Object.keys(data).length > 0;
      setLive(any);
    } catch(err) {}
  };
  es.onerror = function() { setLive(false); };
}

function setLive(on) {
  var dot  = document.getElementById('live-dot');
  var text = document.getElementById('live-text');
  dot.className  = 'live-dot' + (on ? ' on' : '');
  text.className = 'live-text' + (on ? ' on' : '');
  text.textContent = on ? '● LIVE' : 'Offline';
}

function renderStats(data) {
  var el = document.getElementById('stats-list');
  if (!Object.keys(data).length) {
    el.innerHTML = '<div style="color:var(--gray);font-size:11px">Nenhuma câmera ativa</div>';
    return;
  }
  el.innerHTML = '';
  Object.keys(data).forEach(function(idx, slot) {
    var s     = data[idx];
    var color = CAM_COLORS[slot % CAM_COLORS.length];
    var row   = document.createElement('div');
    row.className = 'stat-row';
    row.style.borderLeftColor = color;
    row.innerHTML =
      '<div class="stat-cam" style="color:'+color+'">CAM ' + idx + '</div>' +
      '<div class="stat-vals">' + s.face_count + ' face(s)</div>' +
      '<div class="stat-sub">FPS: ' + s.fps + ' &nbsp;·&nbsp; Engine: ' + s.engine + '</div>';
    el.appendChild(row);
  });
}

// ── Reports ───────────────────────────────────────────────────────────────────
var _rfTimer = null;
var _rfActive = false;

window.loadReports = async function () {
  if (_rfActive) return;
  _rfActive = true;

  try {
    var r    = await fetch('/api/reports/faces');
    allFaces = await r.json();
    renderReportGrid();
    renderReportSummary();
  } catch(e) {
    toast('Erro ao carregar reports: ' + e.message, true);
  } finally {
    _rfActive = false;
    // auto-refresh every 5s while reports tab is visible
    clearTimeout(_rfTimer);
    _rfTimer = setTimeout(function(){
      if (document.getElementById('pane-reports').classList.contains('active'))
        loadReports();
    }, 5000);
  }
};

function renderReportSummary() {
  var el   = document.getElementById('report-summary');
  var n    = allFaces.length;
  var dnn  = allFaces.filter(function(f){ return f.engine === 'DNN'; });
  var avg  = dnn.length ? (dnn.reduce(function(a,f){ return a+f.confidence; },0)/dnn.length) : 0;
  var cams = new Set(allFaces.map(function(f){ return f.camera_index; })).size;
  el.innerHTML =
    card('Total', n, '') +
    card('DNN', dnn.length, 'cy') +
    card('Câmeras', cams || '—', 'cy') +
    card('Conf. Média', avg ? (avg*100).toFixed(0)+'%' : '—', '');
}

function card(lbl, val, cls) {
  return '<div class="sum-card"><div class="sum-lbl">'+lbl+'</div>' +
         '<div class="sum-val '+cls+'">'+val+'</div></div>';
}

['rf-cam','rf-engine','rf-conf'].forEach(function(id){
  document.addEventListener('DOMContentLoaded', function(){
    var el = document.getElementById(id);
    if (el) el.addEventListener('change', renderReportGrid);
  });
});

window.setSort = function (s) {
  faceSort = s;
  ['new','old','conf'].forEach(function(k){
    var b = document.getElementById('s-'+k);
    if (b) b.className = 'sort-btn' + (k === s ? ' active' : '');
  });
  renderReportGrid();
};

function renderReportGrid() {
  var cam  = document.getElementById('rf-cam').value;
  var eng  = document.getElementById('rf-engine').value;
  var minC = parseFloat(document.getElementById('rf-conf').value) || 0;

  // rebuild cam filter
  var camSel = document.getElementById('rf-cam');
  var cur    = camSel.value;
  var cams   = [...new Set(allFaces.map(function(f){ return f.camera_index; }))].sort();
  camSel.innerHTML = '<option value="all">Todas</option>';
  cams.forEach(function(c){
    var o = document.createElement('option');
    o.value = c; o.textContent = 'Câmera ' + c;
    camSel.appendChild(o);
  });
  camSel.value = cur;

  var filtered = allFaces.filter(function(f){
    if (cam !== 'all' && String(f.camera_index) !== String(cam)) return false;
    if (eng !== 'all' && f.engine !== eng) return false;
    if (f.engine === 'DNN' && f.confidence < minC) return false;
    return true;
  });

  if (faceSort === 'new')       filtered.sort(function(a,b){ return b.timestamp.localeCompare(a.timestamp); });
  else if (faceSort === 'old')  filtered.sort(function(a,b){ return a.timestamp.localeCompare(b.timestamp); });
  else                          filtered.sort(function(a,b){ return b.confidence - a.confidence; });

  document.getElementById('r-vis').textContent = filtered.length;
  document.getElementById('r-tot').textContent = allFaces.length;

  var grid = document.getElementById('report-grid');
  grid.innerHTML = '';

  if (!filtered.length) {
    grid.innerHTML =
      '<div class="no-signal" style="grid-column:1/-1">' +
        '<span>◈ SEM DADOS ◈</span>' +
        '<small>Inicie a webcam — faces são capturadas automaticamente</small>' +
      '</div>';
    return;
  }

  filtered.forEach(function(f, i) { grid.appendChild(buildFaceCard(f, i+1)); });
}

function buildFaceCard(f, num) {
  var conf   = f.engine === 'DNN' ? f.confidence : 1.0;
  var cpct   = f.engine === 'DNN' ? (f.confidence*100).toFixed(0)+'%' : 'N/A';
  var bcolor = conf >= .8 ? 'var(--green)' : (conf >= .6 ? 'var(--cyan)' : 'var(--yellow)');
  var ecls   = f.engine === 'DNN' ? 'eng-dnn' : 'eng-haar';
  var card   = document.createElement('div');
  card.className = 'face-card';
  var img = f.image
    ? '<img src="data:image/jpeg;base64,'+f.image+'" loading="lazy" alt="face">'
    : '<div style="aspect-ratio:1;background:#050505;display:flex;align-items:center;justify-content:center;color:#222;font-size:10px">SEM IMG</div>';
  card.innerHTML = img +
    '<div class="info">' +
    '<div class="face-id">FACE #' + String(num).padStart(4,'0') + '</div>' +
    frow('HORÁRIO',   f.timestamp ? f.timestamp.slice(11,19) : '—') +
    frow('DATA',      f.timestamp ? f.timestamp.slice(0,10)  : '—') +
    frow('CÂMERA',    'índice ' + f.camera_index) +
    frow('BBOX',      f.bbox_w + ' × ' + f.bbox_h + ' px') +
    frow('CONFIANÇA', '<span style="color:'+bcolor+'">'+cpct+'</span>') +
    '<div class="conf-bar-wrap"><div class="conf-bar" style="width:'+(conf*100|0)+'%;background:'+bcolor+'"></div></div>' +
    '<span class="eng-badge '+ecls+'">'+f.engine+'</span>' +
    '</div>';
  return card;
}
function frow(k, v) {
  return '<div class="f-row"><span class="f-key">'+k+'</span><span class="f-val">'+v+'</span></div>';
}

window.clearReports = async function () {
  if (!confirm('Limpar todos os registros de faces?')) return;
  await fetch('/api/reports/clear', {method:'POST'});
  allFaces = [];
  renderReportGrid();
  renderReportSummary();
  toast('Registros limpos.');
};

window.pushToGitHub = async function () {
  toast('Publicando no GitHub…');
  try {
    var r    = await fetch('/api/reports/push', {method:'POST'});
    var data = await r.json();
    if (data.ok) {
      toast('✓ Publicado! ' + (data.message || ''));
      if (data.url) {
        setTimeout(function(){ window.open(data.url, '_blank'); }, 800);
      }
    } else {
      toast('✗ ' + (data.error || data.message), true);
    }
  } catch(e) {
    toast('✗ Erro: ' + e.message, true);
  }
};

// ── Users ─────────────────────────────────────────────────────────────────────
window.loadUsers = async function () {
  try {
    var r    = await fetch('/api/users');
    var data = await r.json();
    renderUsersList(data);
  } catch(e) {}
};

function renderUsersList(users) {
  var el = document.getElementById('users-list');
  el.innerHTML = '';
  users.forEach(function(u) {
    var row = document.createElement('div');
    row.className = 'user-row';
    var rcls = u.role === 'master' ? 'role-m' : 'role-u';
    row.innerHTML =
      '<span class="user-role-badge '+rcls+'">'+u.role.toUpperCase()+'</span>' +
      '<span class="user-name">'+u.username+'</span>' +
      '<span class="user-date">'+u.created_at.slice(0,10)+'</span>';
    if (u.role !== 'master') {
      row.innerHTML +=
        '<button class="del-btn" onclick="deleteUser(\''+u.username+'\')">🗑</button>';
    }
    el.appendChild(row);
  });
}

window.createUser = async function () {
  var uname = document.getElementById('new-uname').value.trim();
  var pass  = document.getElementById('new-pass').value;
  var msg   = document.getElementById('user-msg');
  var r = await fetch('/api/users', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({username: uname, password: pass})
  });
  var data = await r.json();
  msg.style.color = data.ok ? 'var(--green)' : 'var(--red)';
  msg.textContent  = data.message;
  if (data.ok) {
    document.getElementById('new-uname').value = '';
    document.getElementById('new-pass').value  = '';
    loadUsers();
  }
};

window.deleteUser = async function (uname) {
  if (!confirm('Deletar usuário "' + uname + '"?')) return;
  var r    = await fetch('/api/users/' + uname, {method:'DELETE'});
  var data = await r.json();
  toast(data.ok ? '✓ ' + data.message : '✗ ' + data.message, !data.ok);
  loadUsers();
};

// ── GitHub settings ───────────────────────────────────────────────────────────
window.loadGitHubConfig = async function () {
  try {
    var r   = await fetch('/api/config/github');
    var cfg = await r.json();
    document.getElementById('cfg-repo').value     = cfg.repo     || '';
    document.getElementById('cfg-branch').value   = cfg.branch   || 'main';
    document.getElementById('cfg-url').value      = cfg.pages_url || '';
    document.getElementById('cfg-auto').checked   = cfg.auto_sync || false;
    document.getElementById('cfg-interval').value = cfg.sync_interval_s || 30;
  } catch(e) {}
};

window.saveGitHub = async function () {
  var msg = document.getElementById('gh-msg');
  var body = {
    token:           document.getElementById('cfg-token').value.trim(),
    repo:            document.getElementById('cfg-repo').value.trim(),
    branch:          document.getElementById('cfg-branch').value.trim() || 'main',
    pages_url:       document.getElementById('cfg-url').value.trim(),
    auto_sync:       document.getElementById('cfg-auto').checked,
    sync_interval_s: parseInt(document.getElementById('cfg-interval').value) || 30,
  };
  var r    = await fetch('/api/config/github', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  var data = await r.json();
  msg.style.color = data.ok ? 'var(--green)' : 'var(--red)';
  msg.textContent  = data.ok ? '✓ Configuração salva.' : '✗ Erro ao salvar.';
};

window.testGitHub = async function () {
  var msg = document.getElementById('gh-msg');
  msg.style.color = 'var(--gray)'; msg.textContent = 'Testando…';
  var r    = await fetch('/api/config/github/test', {method:'POST'});
  var data = await r.json();
  msg.style.color = data.ok ? 'var(--green)' : 'var(--red)';
  msg.textContent  = (data.ok ? '✓ ' : '✗ ') + data.message;
};

// ── Toast ─────────────────────────────────────────────────────────────────────
var _toastTimer = null;
function toast(msg, isError) {
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = 'toast show' + (isError ? ' error' : '');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(function(){
    el.className = 'toast';
  }, 3500);
}
