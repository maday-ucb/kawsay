/* ── Estado global ── */
const state = {
  results: null,
  activeMethod: 'semantic',
  allChunks: [],
  threshold: null,
};

/* ── Utilidades DOM ── */
const $  = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

/* ── Navegación entre vistas ── */
function initNav() {
  $('filter-type').addEventListener('change', renderLibrary);

  document.querySelectorAll('.navigation button').forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      document.querySelectorAll('.navigation button').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      document.querySelectorAll('.view').forEach(v => {
        v.hidden = v.id !== view;
      });
      if (view === 'library') renderLibrary();
      if (view === 'evaluation') loadMetrics();
    });
  });
}

/* ── Cargar estado inicial ── */
async function init() {
  try {
    const health = await fetch('/api/health').then(r => r.json());
    if (health.status !== 'ok') {
      setStatus('⚠ Índice no disponible. Ejecuta build_index y reinicia.');
      return;
    }
    const chunks = await fetch('/api/documents').then(r => r.json());
    state.allChunks = chunks;
    const metricsData = await fetch('/api/metrics').then(r => r.json()).catch(() => null);
    if (metricsData?.threshold_selection?.threshold != null) {
      state.threshold = metricsData.threshold_selection.threshold;
    }
    const docCount = new Set(chunks.map(c => c.document_id)).size;
    $('doc-count').textContent = docCount;
    setStatus(`Listo · ${docCount} documentos · ${chunks.length} fragmentos indexados.`);
    $('ask').disabled = false;
    $('search').disabled = false;
  } catch {
    setStatus('⚠ No se pudo conectar con el servidor.');
  }
}

function setStatus(msg) {
  $('status').textContent = msg;
}

/* ── Formulario principal ── */
function initForm() {
  const textarea = $('query');
  const counter  = $('counter');

  textarea.addEventListener('input', () => {
    counter.textContent = `${textarea.value.length} / 500`;
  });

  // Presets
  document.querySelectorAll('.presets button').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.dataset.question;
      counter.textContent = `${textarea.value.length} / 500`;
      textarea.focus();
    });
  });

  // Solo buscar
  $('search').addEventListener('click', async () => {
    const query = textarea.value.trim();
    if (!query) return;
    await runSearch(query, false);
  });

  // Consultar con fuentes
  $('question-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textarea.value.trim();
    if (!query) return;
    await runSearch(query, true);
  });
}

/* ── Ejecutar búsqueda ── */
async function runSearch(query, withAnswer) {
  setStatus('Buscando…');
  if (withAnswer) setAnswerLoading();
  $('retrieval-panel').hidden = true;
  $('ask').disabled = true;
  $('search').disabled = true;

  try {
    const endpoint = withAnswer ? '/api/ask' : '/api/search';
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    if (res.status === 429) {
      setStatus('⚠ Hay una consulta en proceso. Intenta en unos segundos.');
      clearAnswer();
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(`⚠ Error ${res.status}: ${err.detail || 'consulta inválida'}`);
      clearAnswer();
      return;
    }


    const data = await res.json();
    state.results = data.results ?? data;

    renderRetrieval();
    if (withAnswer) renderAnswer(data);
    else clearAnswer();

    setStatus(`Listo · ${new Set(state.allChunks.map(c => c.document_id)).size} documentos · ${state.allChunks.length} fragmentos indexados.`);
  } catch {
    setStatus('⚠ Error de red. Verifica que el servidor esté corriendo.');
    clearAnswer();
  } finally {

    $('ask').disabled = false;
    $('search').disabled = false;
  }
}

/* ── Renderizar respuesta ── */
function setAnswerLoading() {
  $('answer-panel').hidden = false;
  $('answer-state').className = 'answer-state loading';
  $('answer-state').innerHTML = '<span class="spinner"></span> Procesando…';
  $('answer-text').textContent = '';
  $('evidence').innerHTML = '';
  $('answer-note').textContent = '';
  $('timing').textContent = '';
}

function clearAnswer() {
  $('answer-panel').hidden = true;
  $('answer-state').className = 'answer-state';
  $('answer-state').textContent = '';
}

function renderAnswer(data) {
  const stateEl = $('answer-state');
  const status  = data.status;

  if (status === 'answered') {
    stateEl.className = 'answer-state answered';
    stateEl.textContent = '✓ RESPUESTA CON FUENTES';
  } else if (status === 'abstained') {
    stateEl.className = 'answer-state abstained';
    stateEl.textContent = '— SIN INFORMACIÓN SUFICIENTE';
  } else if (status === 'below_threshold') {
    stateEl.className = 'answer-state below';
    stateEl.textContent = '✗ SIMILITUD INSUFICIENTE';
  } else {
    stateEl.className = 'answer-state abstained';
    stateEl.textContent = '— VERIFICACIÓN FALLIDA';
  }

  $('answer-text').textContent = data.answer || '';

  const evidenceEl = $('evidence');
  evidenceEl.innerHTML = '';
  (data.evidence || []).forEach(ev => {
    const item = el('div', 'evidence-item');
    item.innerHTML = `
      <div class="chunk-id">${ev.chunk_id}</div>
      <blockquote>"${ev.quote}"</blockquote>
    `;
    const btn = el('button', '', 'Ver fuente ↗');
    btn.addEventListener('click', () => scrollToChunk(ev.chunk_id));
    item.appendChild(btn);
    evidenceEl.appendChild(item);
  });

  if (data.total_seconds) {
    $('timing').textContent = `Tiempo total: ${data.total_seconds}s`;
  }

  if (data.gate_passed === false) {
    $('answer-note').textContent =
      `Similitud ${data.results?.semantic?.[0]?.score?.toFixed(3) ?? '—'} por debajo del umbral ${data.threshold?.toFixed(3) ?? '—'}.`;
  }
}

/* ── Renderizar fragmentos recuperados ── */
function renderRetrieval() {
  if (!state.results) return;
  $('retrieval-panel').hidden = false;
  renderCards(state.activeMethod);

  document.querySelectorAll('.method-buttons button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.method-buttons button').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      state.activeMethod = btn.dataset.method;
      renderCards(state.activeMethod);
    });
  });
}

function renderCards(method) {
  const container = $('retrieval-cards');
  container.innerHTML = '';
  const items = state.results?.[method] ?? [];
  const isBm25 = method === 'lexical';

  items.forEach((item, i) => {
    const card = el('div', `retrieval-card rank-${i + 1}`);
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title">${item.heading || item.title}</span>
        <span class="card-score">${item.score?.toFixed(4) ?? '—'}</span>
      </div>
      <div class="card-meta">${item.chunk_id} · ${item.title ?? ''}</div>
      <div class="card-text">${item.text}</div>
    `;
    container.appendChild(card);
  });

  const gateNote = $('gate-note');
  const topScore = items[0]?.score;
  if (topScore !== undefined) {
    const label = isBm25
      ? `Puntaje BM25 más alto: ${topScore.toFixed(2)} (escala no acotada)`
      : `Similitud semántica más alta: ${topScore.toFixed(4)} (escala 0–1)`;
    gateNote.textContent = label;
  }

  // Low-confidence warning for semantic search
  const panel = $('retrieval-panel');
  const existing = panel.querySelector('.low-confidence-banner');
  if (existing) existing.remove();
  if (!isBm25 && topScore !== undefined && topScore < (state.threshold ?? 1)) {
    const banner = el('div', 'low-confidence-banner',
      `⚠ Similitud baja (${topScore.toFixed(3)} ‹ umbral ${state.threshold?.toFixed(3) ?? 'n/d'}). ` +
      `Los resultados son los más cercanos disponibles, pero el corpus probablemente no contiene información sobre esta consulta.`);
    panel.insertBefore(banner, container);
  }
}

/* ── Corpus / Library ── */
function renderLibrary() {
  const container = $('documents');
  container.innerHTML = '';
  const filterType = $('filter-type').value;

  const byDoc = {};
  state.allChunks.forEach(chunk => {
    if (!byDoc[chunk.document_id]) {
      byDoc[chunk.document_id] = { ...chunk, chunks: [] };
    }
    byDoc[chunk.document_id].chunks.push(chunk);
  });

  const docs = Object.values(byDoc).filter(doc => !filterType || doc.doc_type === filterType);

  if (docs.length === 0) {
    container.innerHTML = '<p class="fine">No hay documentos para este tipo.</p>';
  }

  docs.forEach(doc => {
    const item = el('div', 'doc-item');
    item.innerHTML = `
      <div class="doc-header">
        <span class="doc-title">${doc.title}</span>
        <span class="doc-type">${doc.doc_type ?? 'documento'}</span>
      </div>
      <div class="doc-meta">${doc.document_id} · ${doc.chunks.length} fragmento(s) · ${doc.filename}</div>
    `;

    const chunksEl = el('div', 'doc-chunks');
    doc.chunks.forEach(chunk => {
      const ci = el('div', 'chunk-item');
      ci.id = `chunk-${chunk.chunk_id}`;
      ci.innerHTML = `
        <div class="chunk-id-label">${chunk.chunk_id}</div>
        <div class="chunk-heading">${chunk.heading}</div>
        <div class="chunk-text">${chunk.text}</div>
      `;
      chunksEl.appendChild(ci);
    });

    item.appendChild(chunksEl);
    item.addEventListener('click', () => chunksEl.classList.toggle('open'));
    container.appendChild(item);
  });

}

function scrollToChunk(chunkId) {
  document.querySelector('[data-view="library"]').click();
  setTimeout(() => {
    const target = document.getElementById(`chunk-${chunkId}`);
    if (target) {
      target.closest('.doc-chunks')?.classList.add('open');
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.style.outline = `2px solid var(--accent)`;
      setTimeout(() => { target.style.outline = ''; }, 2000);
    } else {
      setStatus(`⚠ Fragmento ${chunkId} no encontrado en el corpus cargado.`);
    }
  }, 100);
}

/* ── Evaluación ── */
async function loadMetrics() {
  try {
    const report = await fetch('/api/metrics').then(r => r.json());
    renderMetrics(report);
  } catch {
    $('retrieval-verdict').textContent = 'No se pudieron cargar las métricas.';
  }
}

function renderMetrics(report) {
  const test = report.partitions?.test;
  if (!test) return;

  // Tarjetas principales
  const cards = $('metric-cards');
  cards.innerHTML = '';
  const semantic = test.retrieval?.semantic ?? {};
  [
    { label: 'NDCG@5 Semántica', value: semantic.ndcg_at_5 },
    { label: 'Recall@5 Semántica', value: semantic.recall_at_5 },
    { label: 'MRR@5 Semántica', value: semantic.mrr_at_5 },
    { label: 'Consultas prueba', value: test.queries, raw: true },
  ].forEach(({ label, value, raw }) => {
    const card = el('div', 'metric-card');
    card.innerHTML = `
      <div class="metric-value">${raw ? value : (value?.toFixed(3) ?? '—')}</div>
      <div class="metric-label">${label}</div>
    `;
    cards.appendChild(card);
  });

  // Tabla comparativa
  const tbody = $('metrics-table');
  tbody.innerHTML = '';
  const lexical = test.retrieval?.lexical ?? {};
  const metricLabels = {
    ndcg_at_5: 'NDCG@5',
    recall_at_5: 'Recall@5',
    precision_at_5: 'Precision@5',
    mrr_at_5: 'MRR@5',
  };
  Object.entries(metricLabels).forEach(([key, label]) => {
    const row = el('tr');
    row.innerHTML = `
      <td>${label}</td>
      <td>${semantic[key]?.toFixed(4) ?? '—'}</td>
      <td>${lexical[key]?.toFixed(4) ?? '—'}</td>
    `;
    tbody.appendChild(row);
  });

  // Veredicto
  const ndcgSem = semantic.ndcg_at_5 ?? 0;
  const ndcgLex = lexical.ndcg_at_5 ?? 0;
  $('retrieval-verdict').textContent = ndcgSem > ndcgLex
    ? `La búsqueda semántica supera a BM25 en NDCG@5 (${ndcgSem.toFixed(3)} vs ${ndcgLex.toFixed(3)}).`
    : `BM25 iguala o supera a la búsqueda semántica en NDCG@5 (${ndcgLex.toFixed(3)} vs ${ndcgSem.toFixed(3)}).`;

  // Métricas del filtro
  const gate = test.gate ?? {};
  if (gate.f1 !== undefined) {
    $('gate-metrics').textContent =
      `Umbral: ${report.threshold_selection?.threshold?.toFixed(4) ?? '—'} · F1: ${gate.f1?.toFixed(3)} · Precision: ${gate.precision?.toFixed(3)} · Recall: ${gate.recall?.toFixed(3)}`;
  }
}

/* ── Arranque ── */
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initForm();
  init();

  // Theme toggle
  const toggle = $('theme-toggle');
  const saved = localStorage.getItem('kawsay-theme');
  if (saved === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    toggle.textContent = '🌙';
  }
  toggle.addEventListener('click', () => {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    if (isLight) {
      document.documentElement.removeAttribute('data-theme');
      toggle.textContent = '☀';
      localStorage.setItem('kawsay-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      toggle.textContent = '🌙';
      localStorage.setItem('kawsay-theme', 'light');
    }
  });
});
