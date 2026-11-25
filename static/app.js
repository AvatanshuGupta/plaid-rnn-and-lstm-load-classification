'use strict';
/* PLAID Load Classifier UI: vanilla JS, no external dependencies. */

// ---------------------------------------------------------------- helpers
const $ = (s, r = document) => r.querySelector(s);
const h = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
};
const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const c of kids.flat(Infinity)) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
};
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const f3 = v => (v == null ? '–' : typeof v === 'number' ? v.toFixed(3) : String(v));
const f4 = v => (v == null ? '–' : v.toFixed(4));
const pct = v => (v == null ? '–' : (100 * v).toFixed(1) + '%');
const COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c'];

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { const j = await r.json(); m = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(m);
  }
  return r.json();
}
function showError(msg) {
  const b = $('#error');
  b.textContent = msg; b.classList.remove('hidden');
  clearTimeout(showError.t); showError.t = setTimeout(() => b.classList.add('hidden'), 9000);
}
const card = (title, sub, ...kids) =>
  h('section', { class: 'card' }, title && h('h2', {}, title), sub && h('p', { class: 'sub' }, sub), ...kids);
const stat = (l, v, s) => h('div', { class: 'stat' }, h('div', { class: 'l' }, l), h('div', { class: 'v' }, v), s && h('div', { class: 's' }, s));
const notice = html => h('div', { class: 'notice', html });
function pills(options, current, onPick) {
  return h('div', { class: 'pills' }, options.map(o =>
    h('button', { class: 'pill' + (o === current ? ' on' : ''), onclick: () => onPick(o) }, o)));
}
function simpleTable(head, rows, numFrom = 1) {
  return h('div', { class: 'tablewrap' }, h('table', {},
    h('thead', {}, h('tr', {}, head.map((c, i) => h('th', { class: i >= numFrom ? 'num' : '' }, c)))),
    h('tbody', {}, rows.map(r => h('tr', {}, r.map((c, i) => h('td', { class: i >= numFrom ? 'num' : '' }, c)))))));
}

// ---------------------------------------------------------------- charts
function hbar(items, { width = 780, rowH = 24, labelW = 230, fmt = f4, signed = false } = {}) {
  const H = items.length * rowH + 8, plotW = width - labelW - 80;
  const mx = Math.max(1e-12, ...items.map(i => Math.abs(i.value)));
  const zero = signed ? labelW + plotW / 2 : labelW, sc = (signed ? plotW / 2 : plotW) / mx;
  const s = svg('svg', { viewBox: `0 0 ${width} ${H}`, class: 'chart' });
  items.forEach((it, k) => {
    const y = k * rowH + 4, w = Math.abs(it.value) * sc, pos = it.value >= 0, x = pos ? zero : zero - w;
    s.append(svg('text', { x: labelW - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end' }, it.label));
    s.append(svg('rect', { x, y: y + 3, width: Math.max(w, 1), height: rowH - 8, rx: 3,
      style: `fill:${it.color || (pos ? 'var(--accent)' : 'var(--neg)')}` }, svg('title', {}, `${it.label}: ${fmt(it.value)}`)));
    s.append(svg('text', { x: pos ? x + w + 6 : x - 6, y: y + rowH / 2 + 4, 'text-anchor': pos ? 'start' : 'end', class: 'val' }, fmt(it.value)));
  });
  if (signed) s.append(svg('line', { x1: zero, x2: zero, y1: 0, y2: H, class: 'grid' }));
  return s;
}

function lineChart(xs, series, { width = 520, height = 240, title = '', fmt = v => v.toFixed(2), xlabel = '' } = {}) {
  const m = { l: 48, r: 12, t: 30, b: 34 }, W = width - m.l - m.r, H = height - m.t - m.b;
  const all = series.flatMap(s => s.ys.filter(v => v != null && isFinite(v)));
  if (!all.length) return h('div', { class: 'muted' }, 'no data');
  let lo = Math.min(...all), hi = Math.max(...all);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * .06; lo -= pad; hi += pad;
  const X = i => m.l + (xs.length <= 1 ? 0 : i / (xs.length - 1)) * W, Y = v => m.t + H - (v - lo) / (hi - lo) * H;
  const s = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart' });
  s.append(svg('text', { x: m.l, y: 16, class: 'ttl' }, title));
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    s.append(svg('line', { x1: m.l, x2: m.l + W, y1: y, y2: y, class: 'grid' }), svg('text', { x: m.l - 6, y: y + 4, 'text-anchor': 'end', class: 'tick' }, fmt(v)));
  }
  const nt = Math.min(xs.length, 6);
  for (let k = 0; k < nt; k++) {
    const i = Math.round(k * (xs.length - 1) / Math.max(nt - 1, 1));
    s.append(svg('text', { x: X(i), y: height - 14, 'text-anchor': 'middle', class: 'tick' }, xs[i]));
  }
  if (xlabel) s.append(svg('text', { x: m.l + W / 2, y: height - 1, 'text-anchor': 'middle', class: 'tick' }, xlabel));
  series.forEach((se, k) => {
    const pts = se.ys.map((v, i) => (v == null ? null : `${X(i)},${Y(v)}`)).filter(Boolean).join(' ');
    s.append(svg('polyline', { points: pts, class: 'line', style: `stroke:${se.color || COLORS[k]}` + (se.dash ? ';stroke-dasharray:5 3' : '') }));
    if (series.length > 1) {
      s.append(svg('rect', { x: width - 12 - 90 * (series.length - k), y: 8, width: 10, height: 10, rx: 2, style: `fill:${se.color || COLORS[k]}` }),
        svg('text', { x: width - 12 - 90 * (series.length - k) + 14, y: 17, class: 'tick' }, se.name));
    }
  });
  return s;
}

// matrix as a coloured grid; M[row][col]
function heat(rows, cols, M, { fmt = v => v.toFixed(2), diverge = false, max = null, showVals = true } = {}) {
  const mx = max ?? Math.max(1e-12, ...M.flat().map(Math.abs));
  const t = h('table', { class: 'heat' });
  t.append(h('thead', {}, h('tr', {}, h('th'), cols.map(c => h('th', { class: 'rot' }, h('div', {}, c))))));
  const body = h('tbody');
  M.forEach((row, i) => body.append(h('tr', {}, h('th', { class: 'rowh' }, rows[i]),
    row.map((v, j) => {
      const a = Math.min(1, Math.abs(v) / mx), rgb = diverge && v < 0 ? '37,99,235' : diverge ? '220,38,38' : '37,99,235';
      return h('td', { style: `background:rgba(${rgb},${a.toFixed(3)});color:${a > .55 ? '#fff' : 'inherit'}`, title: `${rows[i]} × ${cols[j]}: ${fmt(v)}` },
        showVals && v !== 0 ? fmt(v) : '');
    }))));
  t.append(body);
  return h('div', { class: 'tablewrap' }, t);
}

// tiny markdown renderer for report.md
function md2html(md) {
  const inline = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>');
  const L = md.split('\n'), out = []; let i = 0;
  while (i < L.length) {
    const l = L[i];
    if (/^#{1,6} /.test(l)) { const n = l.match(/^#+/)[0].length; out.push(`<h${n}>${inline(l.replace(/^#+ /, ''))}</h${n}>`); i++; }
    else if (l.startsWith('|')) {
      const rows = []; while (i < L.length && L[i].startsWith('|')) rows.push(L[i++].replace('|SHAP|', '¦SHAP¦'));
      const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const head = cells(rows[0]), body = rows.slice(2).map(cells);
      out.push('<div class="tablewrap"><table><thead><tr>' + head.map(c => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>' +
        body.map(r => '<tr>' + r.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>');
    } else if (/^!\[.*\]\(.*\)/.test(l.trim())) {
      const m = l.trim().match(/^!\[(.*)\]\((.*)\)/); out.push(`<p><img src="/results/${esc(m[2])}" alt="${esc(m[1])}"></p>`); i++;
    } else if (/^- /.test(l)) {
      const items = []; while (i < L.length && /^- /.test(L[i])) items.push(L[i++].slice(2));
      out.push('<ul>' + items.map(x => `<li>${inline(x)}</li>`).join('') + '</ul>');
    } else if (!l.trim()) i++;
    else { const p = []; while (i < L.length && L[i].trim() && !/^(#|\||!\[|- )/.test(L[i])) p.push(L[i++]); out.push(`<p>${inline(p.join(' '))}</p>`); }
  }
  return out.join('\n');
}

// ---------------------------------------------------------------- state
const S = { info: null, report: null, perf: null, shapModel: null, shap: {}, shapClass: 'overall', topN: 20, records: null, pred: {} };
async function getReport() {
  if (S.report === undefined || S.report === null) {
    try { S.report = await api('/api/report'); } catch (e) { S.report = false; S.reportErr = e.message; }
  }
  return S.report;
}
const classesOf = R => (R && R.report.split ? Object.keys(R.report.split.per_class) : S.info.classes);
const NEED_REPORT = () => notice('<b>No report found.</b> Run <code>python train_sequence_models.py --features features/all_features.csv --out_dir results --shap</code> first. ' + esc(S.reportErr || ''));

// ---------------------------------------------------------------- OVERVIEW
async function viewOverview(root) {
  const R = await getReport();
  if (!R) return root.append(NEED_REPORT());
  const { info, split, run_info, results } = R.report;
  root.append(h('div', { class: 'stats' },
    stat('Records', info.records), stat('Classes', info.classes), stat('Features', info['feature count used']),
    stat('Sequence length', info['sequence length (windows)'], 'windows per sample'),
    ...Object.entries(results).filter(([k]) => !k.includes('baseline')).map(([k, m]) => stat(k + ' accuracy', pct(m.accuracy), 'macro-F1 ' + f3(m.macro_f1)))));

  // model comparison
  const names = Object.keys(results);
  const keys = [['accuracy', 1], ['balanced_accuracy', 1], ['macro_f1', 1], ['weighted_f1', 1], ['macro_precision', 1], ['macro_recall', 1],
    ['cohen_kappa', 1], ['mcc', 1], ['top3_accuracy', 1], ['roc_auc_macro_ovr', 1], ['log_loss', -1]];
  const rows = keys.filter(([k]) => names.some(n => results[n][k] != null)).map(([k, dir]) => {
    const vals = names.map(n => results[n][k]), best = dir > 0 ? Math.max(...vals.filter(v => v != null)) : Math.min(...vals.filter(v => v != null));
    return [k.replace(/_/g, ' '), ...vals.map(v => h('span', { style: v === best ? 'font-weight:700;color:var(--accent)' : '' }, f4(v)))];
  });
  rows.push(['accuracy 95% CI', ...names.map(n => `[${f3(results[n].accuracy_ci95[0])}, ${f3(results[n].accuracy_ci95[1])}]`)]);
  rows.push(['macro-F1 95% CI', ...names.map(n => `[${f3(results[n].macro_f1_ci95[0])}, ${f3(results[n].macro_f1_ci95[1])}]`)]);
  root.append(card('Model comparison', 'Held-out test records, record-level (chunk probabilities averaged per record). Best value per row in blue.', simpleTable(['metric', ...names], rows)));

  root.append(h('div', { class: 'grid g2' },
    card('Training summary', null, simpleTable(['model', 'params', 'epochs', 'best epoch', 'time (s)', 'chunk acc', 'chunk F1'],
      Object.entries(run_info).map(([n, r]) => [n, r.n_params ?? '–', r.epochs_run ?? '–', r.best_epoch ?? '–', r.train_time_s != null ? r.train_time_s.toFixed(1) : '–', f3(r.chunk_acc), f3(r.chunk_f1)]))),
    card('Data split', 'Split by record id (never by window), stratified by class.', simpleTable(['class', 'train', 'val', 'test'],
      [...Object.entries(split.per_class).map(([c, v]) => [c, ...v]), [h('b', {}, 'total'), ...split.totals.map(v => h('b', {}, v))]]))));

  root.append(card('Configuration', null, simpleTable(['setting', 'value'], Object.entries(info).map(([k, v]) => [k, String(v)]), 99)));
  const imgs = ['per_class_f1.png', 'curves.png'].filter(n => S.info.images.includes(n));
  if (imgs.length) root.append(card('Summary plots', null, imgs.map(n => h('p', {}, h('img', { class: 'plot', src: '/results/' + n })))));
}

// ---------------------------------------------------------------- PERFORMANCE
async function viewPerformance(root) {
  const R = await getReport();
  if (!R) return root.append(NEED_REPORT());
  const { results, history, run_info } = R.report, names = Object.keys(results);
  if (!S.perf || !results[S.perf]) S.perf = names.find(n => !n.includes('baseline')) || names[0];
  const redraw = () => { root.innerHTML = ''; viewPerformance(root); };
  root.append(pills(names, S.perf, n => { S.perf = n; redraw(); }));
  const m = results[S.perf], classes = classesOf(R), ri = run_info[S.perf] || {};

  root.append(h('div', { class: 'stats' },
    stat('Accuracy', pct(m.accuracy), `95% CI ${pct(m.accuracy_ci95[0])} – ${pct(m.accuracy_ci95[1])}`),
    stat('Macro-F1', f3(m.macro_f1), `95% CI ${f3(m.macro_f1_ci95[0])} – ${f3(m.macro_f1_ci95[1])}`),
    stat('Balanced acc.', f3(m.balanced_accuracy)), stat('Weighted F1', f3(m.weighted_f1)),
    stat('Cohen κ', f3(m.cohen_kappa)), stat('MCC', f3(m.mcc)), stat('Log loss', f3(m.log_loss)),
    m.top3_accuracy != null && stat('Top-3 acc.', pct(m.top3_accuracy)), m.roc_auc_macro_ovr != null && stat('ROC-AUC (OvR)', f3(m.roc_auc_macro_ovr)),
    ri.chunk_acc != null && stat('Chunk-level acc.', pct(ri.chunk_acc), 'before record averaging')));

  // per-class table
  root.append(card('Per-class results', 'Precision, recall and F1 on the test records; support = number of test records.',
    h('div', { class: 'tablewrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['class', 'precision', 'recall', 'F1', 'support', 'AUC (OvR)'].map((c, i) => h('th', { class: i ? 'num' : '' }, c)))),
      h('tbody', {}, classes.map(c => {
        const r = m.per_class[c], auc = (m.roc_auc_per_class || {})[c];
        return h('tr', {}, h('td', {}, c), h('td', { class: 'num' }, f3(r.precision)), h('td', { class: 'num' }, f3(r.recall)),
          h('td', { class: 'num' }, h('span', { class: 'bar', style: `width:${Math.round(70 * r['f1-score'])}px` }), f3(r['f1-score'])),
          h('td', { class: 'num' }, r.support), h('td', { class: 'num' }, f3(auc)));
      }))))));

  // confusion matrix
  const cm = m.confusion_matrix, norm = cm.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s); });
  let mode = 'norm';
  const cmBox = h('div');
  const drawCM = () => { cmBox.innerHTML = ''; cmBox.append(mode === 'norm' ? heat(classes.map(c => 'true: ' + c), classes.map(c => 'pred: ' + c), norm, { max: 1, fmt: v => (100 * v).toFixed(0) + '%' })
    : heat(classes.map(c => 'true: ' + c), classes.map(c => 'pred: ' + c), cm, { fmt: v => String(v) })); };
  drawCM();
  root.append(card('Confusion matrix', 'Rows = true class, columns = predicted class.',
    h('div', { class: 'pills' }, h('button', { class: 'pill on', onclick: e => { mode = 'norm'; [...e.target.parentNode.children].forEach(b => b.classList.toggle('on', b === e.target)); drawCM(); } }, 'row-normalised %'),
      h('button', { class: 'pill', onclick: e => { mode = 'count'; [...e.target.parentNode.children].forEach(b => b.classList.toggle('on', b === e.target)); drawCM(); } }, 'counts')),
    cmBox,
    m.top_confusions && m.top_confusions.length ? h('div', {}, h('h3', {}, 'Most frequent confusions'),
      simpleTable(['true', 'predicted', 'count'], m.top_confusions.map(c => [c[0], c[1], c[2]]), 2)) : h('p', { class: 'muted' }, 'No misclassifications on the test set.')));

  // training curves
  const hs = history && history[S.perf];
  if (hs) {
    const best = ri.best_epoch;
    root.append(card('Training curves', `Best epoch ${best} (lowest validation loss, early stopping). Validation metrics are record-level on the validation split.`,
      h('div', { class: 'grid g2' },
        lineChart(hs.epoch, [{ name: 'train', ys: hs.train_loss }, { name: 'val', ys: hs.val_loss }], { title: 'Loss', xlabel: 'epoch', fmt: v => v.toFixed(2) }),
        lineChart(hs.epoch, [{ name: 'accuracy', ys: hs.val_acc }, { name: 'macro-F1', ys: hs.val_macro_f1 }], { title: 'Validation accuracy / macro-F1', xlabel: 'epoch' }),
        lineChart(hs.epoch, [{ name: 'lr', ys: hs.lr }], { title: 'Learning rate', xlabel: 'epoch', fmt: v => v.toExponential(0) }))));
  } else if (S.perf.includes('baseline')) {
    root.append(notice('The RandomForest baseline uses per-record mean/std/min/max of the same features. If the sequence models do not beat it, the temporal structure is adding little.'));
  }
}

// ---------------------------------------------------------------- SHAP
async function viewShap(root) {
  const models = S.info.models.filter(m => m.has_shap).map(m => m.name);
  if (!models.length) return root.append(notice('<b>No SHAP results found.</b> Train with <code>--shap</code>: <code>python train_sequence_models.py --features features/all_features.csv --out_dir results --shap</code>'));
  if (!models.includes(S.shapModel)) S.shapModel = models.includes('rnn') ? 'rnn' : models[0];
  const redraw = () => { root.innerHTML = ''; viewShap(root); };
  root.append(pills(models.map(m => m.toUpperCase()), S.shapModel.toUpperCase(), n => { S.shapModel = n.toLowerCase(); redraw(); }));
  let D = S.shap[S.shapModel];
  if (!D) D = S.shap[S.shapModel] = await api('/api/shap/' + S.shapModel);

  root.append(notice(`SHAP values (GradientExplainer / expected gradients) are in <b>logit units</b> on standardised features. A feature's importance is the absolute value of its SHAP sum over the time steps of a sequence, averaged over the explained test chunks. ` +
    (D.additivity_error != null ? `Additivity error: <b>${D.additivity_error.toFixed(3)}</b> logits (lower is better; raise <code>--shap_nsamples</code> to reduce). ` : '') +
    `Many features are strongly correlated (power, current RMS, ...), so SHAP splits credit among them: judge feature <i>groups</i> rather than single columns.`));
  root.append(h('div', { style: 'height:14px' }));

  const cols = ['overall', ...D.classes];
  const valuesFor = c => (c === 'overall' ? D.overall : D.matrix.map(r => r[D.classes.indexOf(c)]));
  const vals = valuesFor(S.shapClass), order = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]);

  root.append(card('Global feature importance', 'Pick a class to see which features drive that appliance (or "overall").',
    h('div', { class: 'row' },
      h('label', { class: 'f' }, 'Class', h('select', { onchange: e => { S.shapClass = e.target.value; redraw(); } }, cols.map(c => h('option', { value: c, selected: c === S.shapClass }, c)))),
      h('label', { class: 'f' }, `Top ${S.topN} features`, h('input', { type: 'range', min: 5, max: Math.min(40, D.features.length), value: S.topN, oninput: e => { S.topN = +e.target.value; redraw(); } }))),
    hbar(order.slice(0, S.topN).map(i => ({ label: D.features[i], value: vals[i] }))),
    h('p', {}, h('a', { href: D.csv, download: '' }, '⬇ download full table (CSV)'))));

  const topK = D.features.slice(0, 20);
  root.append(card('Class × feature heat-map', 'Mean |SHAP| per class for the 20 most important features. Darker = stronger influence on that class.',
    heat(topK, D.classes, topK.map((f, i) => D.matrix[i]), { fmt: v => v.toFixed(2) })));

  const blocks = [];
  if (D.temporal.length) blocks.push(card('Temporal importance', 'Total absolute attribution at each position inside the input sequence (0 = oldest window). Shows which part of the recording the model relies on, e.g. the switch-on transient.',
    lineChart(D.temporal.map((_, i) => i), [{ name: 'attribution', ys: D.temporal }], { title: 'Sum of |SHAP| over features', xlabel: 'time step (window) in sequence', fmt: v => v.toFixed(2), width: 600, height: 260 })));
  const bee = S.shapClass !== 'overall' ? D.beeswarm[S.shapClass] : null;
  blocks.push(card('Beeswarm (per class)', 'Each dot is a test sequence; x = its SHAP contribution to the class logit, colour = feature value (red high, blue low).',
    S.shapClass === 'overall' ? h('p', { class: 'muted' }, 'Select a class above to see its beeswarm plot.') : bee ? h('img', { class: 'plot', src: bee }) : h('p', { class: 'muted' }, 'No beeswarm image for this class.')));
  root.append(h('div', { class: 'grid g2' }, blocks));

  // searchable table
  const tbody = h('tbody'), q = h('input', { placeholder: 'filter features…', oninput: () => fill() });
  const fill = () => { tbody.innerHTML = ''; D.features.forEach((f, i) => { if (f.toLowerCase().includes(q.value.toLowerCase())) tbody.append(h('tr', {}, h('td', {}, i + 1), h('td', {}, f), h('td', { class: 'num' }, f4(D.overall[i])), D.classes.map((c, j) => h('td', { class: 'num' }, D.matrix[i][j].toFixed(3))))); }); };
  fill();
  root.append(card('All features', null, q, h('div', { class: 'tablewrap', style: 'max-height:420px;overflow:auto;margin-top:10px' },
    h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'feature'), h('th', { class: 'num' }, 'overall'), D.classes.map(c => h('th', { class: 'num' }, c)))), tbody))));
}

// ---------------------------------------------------------------- PREDICT
async function viewPredict(root) {
  const models = S.info.models.map(m => m.name);
  if (!models.length) return root.append(notice('<b>No trained model (.pt) found</b> in ' + esc(S.info.results_dir)));
  if (!S.records) { try { S.records = await api('/api/records'); } catch (e) { S.records = []; } }
  let src = 'record';
  const inp = {
    model: h('select', {}, models.map(m => h('option', { value: m, selected: m === S.info.default_model }, m.toUpperCase()))),
    rec: h('input', { list: 'reclist', placeholder: 'type or pick a record id', style: 'width:260px', value: S.lastRec || '' }),
    file: h('input', { type: 'file', accept: '.csv' }),
    window: h('input', { type: 'number', placeholder: `auto (${S.info.window ?? 500})`, style: 'width:120px' }),
    step: h('input', { type: 'number', placeholder: '= window', style: 'width:100px' }),
    fs: h('input', { type: 'number', value: 30000, style: 'width:100px' }),
    order: h('select', {}, h('option', { value: 'iv' }, 'current, voltage'), h('option', { value: 'vi' }, 'voltage, current')),
    f0: h('input', { type: 'number', placeholder: 'auto', style: 'width:90px' }),
  };
  const panels = {
    record: h('div', { class: 'row' }, h('label', { class: 'f' }, 'Record id', inp.rec),
      h('datalist', { id: 'reclist' }, S.records.map(r => h('option', { value: r.id }, `${r.label}${r.in_test ? ' (test)' : ''}`))),
      h('span', { class: 'muted', style: 'align-self:center' }, S.records.length ? `${S.records.length} records in all_features.csv. Records tagged "test" were held out from training.` : 'all_features.csv not found.')),
    features: h('div', { class: 'row' }, h('label', { class: 'f' }, 'Features CSV (same columns as all_features.csv, one row per window)', inp.file)),
    raw: h('div', { class: 'row' }, h('label', { class: 'f' }, 'Raw current/voltage CSV', inp.file.cloneNode(true)),
      h('label', { class: 'f' }, 'Column order', inp.order), h('label', { class: 'f' }, 'Window (samples)', inp.window),
      h('label', { class: 'f' }, 'Step', inp.step), h('label', { class: 'f' }, 'fs (Hz)', inp.fs), h('label', { class: 'f' }, 'Mains f0 (Hz)', inp.f0)),
  };
  // ---- manual entry panel (fields come from the model's own feature list)
  const manualBox = h('div', { class: 'manual' }, h('p', { class: 'muted' }, 'Loading…'));
  panels.manual = manualBox;
  let spec = null, specModel = null;
  const fmtNum = v => { if (v == null) return ''; const a = Math.abs(v); return a !== 0 && (a < 0.01 || a >= 1e5) ? v.toExponential(2) : String(+v.toPrecision(4)); };
  const buildManual = () => {
    manualBox.innerHTML = '';
    manualBox.append(notice('Type whatever you know; <b>leave the rest blank</b>. Dependent quantities are derived automatically when possible ' +
      '(e.g. S, pf, Q, Z, R from <code>v_rms</code>, <code>i_rms</code>, <code>P_real_W</code>; cos φ, P1, Q1, X1 … from <code>phase_deg</code>), ' +
      'and everything else is set to the training median shown as placeholder. The values describe <b>one steady-state window</b>, which is repeated to fill the model\'s input sequence, ' +
      'so switch-on transients cannot be represented here (use a record or an upload for that). Hover a field for its meaning and training range.'));
    const field = f => h('label', { class: 'f', title: `${f.desc}\ntraining range (0.5–99.5 %): ${fmtNum(f.lo)} … ${fmtNum(f.hi)}` },
      f.name + (f.unit ? ` [${f.unit}]` : ''), h('input', { type: 'number', step: 'any', 'data-f': f.name, placeholder: 'typical ' + fmtNum(f.median), style: 'width:150px' }));
    const byName = Object.fromEntries(spec.features.map(f => [f.name, f]));
    manualBox.append(h('h3', {}, 'Key measurements'), h('div', { class: 'row' }, spec.key.map(k => field(byName[k]))));
    const groups = {};
    spec.features.filter(f => !spec.key.includes(f.name)).forEach(f => { (groups[f.group] = groups[f.group] || []).push(f); });
    for (const [g, list] of Object.entries(groups))
      manualBox.append(h('details', {}, h('summary', {}, `${g} (${list.length})`), h('div', { class: 'row' }, list.map(field))));
    manualBox.append(h('div', { class: 'row' }, h('button', { class: 'btn alt', onclick: () => manualBox.querySelectorAll('input').forEach(i => { i.value = ''; }) }, 'Clear all')));
  };
  const loadSpec = async () => {
    if (spec && specModel === inp.model.value) return;
    spec = await api('/api/feature_spec?model=' + inp.model.value); specModel = inp.model.value; buildManual();
  };
  inp.model.addEventListener('change', () => { if (src === 'manual') loadSpec().catch(e => showError(e.message)); });
  const holder = h('div'), out = h('div'), expl = h('div');
  const setSrc = s => { src = s; holder.innerHTML = ''; holder.append(panels[s]); if (s === 'manual') loadSpec().catch(e => showError(e.message)); [...radios.children].forEach(b => b.classList.toggle('on', b.dataset.s === s)); };
  const radios = h('div', { class: 'pills' }, [['record', 'Existing record'], ['manual', 'Enter values manually'], ['features', 'Upload features CSV'], ['raw', 'Upload raw I/V CSV']]
    .map(([s, t]) => h('button', { class: 'pill', 'data-s': s, onclick: () => setSrc(s) }, t)));

  const fd = (extra = {}) => {
    const f = new FormData(); f.append('model', inp.model.value); f.append('source', src);
    f.append('record_id', inp.rec.value.trim()); S.lastRec = inp.rec.value.trim();
    if (src === 'raw' || src === 'features') { const fileEl = (src === 'raw' ? panels.raw : panels.features).querySelector('input[type=file]'); if (fileEl.files[0]) f.append('file', fileEl.files[0]); }
    if (src === 'manual') { const o = {}; manualBox.querySelectorAll('input[data-f]').forEach(i => { if (i.value !== '') o[i.dataset.f] = i.value; }); f.append('manual', JSON.stringify(o)); }
    f.append('window', inp.window.value || 0); f.append('step', inp.step.value || 0); f.append('fs', inp.fs.value || 30000);
    f.append('order', inp.order.value); f.append('f0', inp.f0.value || 0);
    for (const [k, v] of Object.entries(extra)) f.append(k, v);
    return f;
  };
  const btnP = h('button', { class: 'btn', onclick: async () => {
    btnP.disabled = true; out.innerHTML = ''; expl.innerHTML = '';
    try { const r = await api('/api/predict', { method: 'POST', body: fd() }); S.pred = r; renderPrediction(out, expl, r, fd); } catch (e) { showError(e.message); }
    btnP.disabled = false; } }, 'Predict');

  root.append(card('Classify a recording', `Predicting the “${(S.info.target || 'class').replace(/_/g, ' ')}” the models were trained on. Choose the model and the input; recordings are cut into sequences of windows exactly as in training and the per-sequence probabilities are averaged.`,
    h('div', { class: 'row' }, h('label', { class: 'f' }, 'Model', inp.model)), radios, holder, h('div', { class: 'row' }, btnP)), out, expl);
  setSrc('record');
}

function renderPrediction(out, expl, r, fd) {
  const tgt = (r.target || 'class').replace(/_/g, ' '), mm = r.meta.manual;
  out.append(card('Prediction', null,
    h('div', { class: 'muted', style: 'font-size:13px' }, `predicted ${tgt}`),
    h('div', { class: 'verdict' }, r.prediction, ' ', h('span', { class: 'muted', style: 'font-size:18px' }, pct(r.confidence))),
    h('p', {}, h('span', { class: 'tag' }, r.model.toUpperCase()), ' ',
      r.true_label && r.correct != null ? h('span', { class: 'tag ' + (r.correct ? 'good' : 'bad') }, `true ${tgt}: ${r.true_label} ${r.correct ? '✓' : '✗'}`) : null,
      r.true_label && r.correct == null ? h('span', { class: 'tag' }, `true ${tgt}: ${r.true_label} (not a class the model was trained on)`) : null, ' ',
      r.meta.in_test === false ? h('span', { class: 'tag warn' }, 'record was in train/val data, accuracy on it is optimistic') : null, ' ',
      mm ? h('span', { class: 'tag warn', title: mm.derived.length ? 'derived: ' + mm.derived.join(', ') : '' }, `manual input: ${mm.entered} entered, ${mm.derived.length} derived, ${mm.defaulted} set to training median`) : null, ' ',
      h('span', { class: 'muted' }, mm ? 'one steady-state window, repeated to fill the sequence' : `${r.n_windows} windows → ${r.n_chunks} sequences of ${r.seq_len}` + (r.meta.window ? ` (window ${r.meta.window}, ${r.meta.samples} samples)` : ''))),
    h('h3', {}, `Probability of each ${tgt} class (record level)`),
    hbar(r.ranking.slice(0, 10).map((x, i) => ({ label: x.class, value: x.prob, color: i === 0 ? 'var(--accent)' : 'var(--muted)' })), { fmt: v => pct(v), labelW: 170 }),
    h('h3', {}, 'Per-sequence probabilities over the recording'),
    h('p', { class: 'sub' }, 'Each column is one sequence (time runs left to right). A stable dark row means the model is consistent; changes show transients that look like other loads.'),
    heat(r.ranking.slice(0, 8).map(x => x.class), r.chunks.map(c => 'w' + c.start_window),
      r.ranking.slice(0, 8).map(x => r.chunks.map(c => c.probs[r.classes.indexOf(x.class)])), { max: 1, fmt: v => (100 * v).toFixed(0), showVals: r.chunks.length <= 12 })));

  const sel = h('select', {}, r.classes.map(c => h('option', { value: c, selected: c === r.prediction }, c)));
  const btnE = h('button', { class: 'btn alt', onclick: async () => {
    btnE.disabled = true; btnE.textContent = 'Computing SHAP…'; expl.innerHTML = '';
    try { const e = await api('/api/explain', { method: 'POST', body: fd({ target_class: sel.value }) }); renderExplanation(expl, e); } catch (e) { showError(e.message); }
    btnE.disabled = false; btnE.textContent = 'Explain with SHAP'; } }, 'Explain with SHAP');
  out.append(card('Why this prediction?', 'Local SHAP explanation of one recording for a chosen class. Takes a few seconds.', h('div', { class: 'row' }, h('label', { class: 'f' }, 'Explain class', sel), btnE)));
}

function renderExplanation(root, e) {
  root.append(card(`SHAP explanation for class “${e.target_class}”`,
    `Averaged over ${e.n_chunks_explained} sequences. Baseline = the average training window (logit ${e.baseline_logit.toFixed(2)}); model output for this recording = logit ${e.output_logit.toFixed(2)}; sum of SHAP values ≈ ${e.shap_sum.toFixed(2)}. Red pushes towards the class, blue pushes away.`,
    h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--accent)' }), 'increases class score'), h('span', {}, h('i', { style: 'background:var(--neg)' }), 'decreases class score')),
    hbar(e.features.map(f => ({ label: `${f.feature} (z=${f.value_z.toFixed(1)})`, value: f.shap, color: f.shap >= 0 ? 'var(--accent)' : 'var(--neg)' })), { signed: true, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(3), labelW: 260 }),
    h('p', { class: 'sub' }, 'z = the feature\'s average standardised value in this recording (0 = typical training value).')));
  root.append(h('div', { class: 'grid g2' },
    card('Contribution over time', 'Signed sum of SHAP values per window position inside the sequence.',
      lineChart(e.per_timestep.map((_, i) => i), [{ name: 'shap', ys: e.per_timestep }], { title: 'Net contribution', xlabel: 'time step (window)', fmt: v => v.toFixed(2), width: 520 })),
    card('Top features over time', 'Rows = most influential features, columns = position in the sequence.',
      heat(e.heat_features, e.heat[0].map((_, i) => 't' + i), e.heat, { diverge: true, fmt: v => v.toFixed(2), showVals: false }))));
}

// ---------------------------------------------------------------- REPORT
async function viewReport(root) {
  const R = await getReport();
  if (!R) return root.append(NEED_REPORT());
  root.append(card(null, null, h('p', {}, h('a', { href: '/results/report.md', download: '' }, '⬇ report.md'), ' · ', h('a', { href: '/results/report.json', download: '' }, '⬇ report.json')),
    h('div', { class: 'md', html: md2html(R.md) })));
}

// ---------------------------------------------------------------- router
const TABS = [['overview', 'Overview', viewOverview], ['performance', 'Performance', viewPerformance], ['shap', 'SHAP analysis', viewShap],
  ['predict', 'Predict', viewPredict], ['report', 'Full report', viewReport]];
async function route() {
  const t = (location.hash || '#overview').slice(1), tab = TABS.find(x => x[0] === t) || TABS[0];
  $('#nav').innerHTML = '';
  TABS.forEach(x => $('#nav').append(h('a', { href: '#' + x[0], class: x[0] === tab[0] ? 'active' : '' }, x[1])));
  const root = $('#view'); root.innerHTML = '';
  try { await tab[2](root); } catch (e) { root.append(notice('<b>Error:</b> ' + esc(e.message))); console.error(e); }
}
(async function boot() {
  try {
    S.info = await api('/api/info');
    $('#status').textContent = S.info.models.map(m => m.name.toUpperCase()).join(' · ');
  } catch (e) { $('#view').innerHTML = ''; return showError('Cannot reach API: ' + e.message); }
  window.addEventListener('hashchange', route);
  route();
})();