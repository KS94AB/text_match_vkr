const fileInput = document.getElementById('file-input');
const groundTruthInput = document.getElementById('ground-truth-input');
const selectedFiles = document.getElementById('selected-files');
const selectedGroundTruth = document.getElementById('selected-ground-truth');
const methodInput = document.getElementById('method');
const methodGrid = document.getElementById('method-grid');
const form = document.getElementById('analysis-form');
const resultsRoot = document.getElementById('results-root');
const statusBanner = document.getElementById('status-banner');
const clearButton = document.getElementById('clear-button');

const methodDetails = window.__METHOD_DETAILS__ || {};
const defaultMethod = window.__DEFAULT_METHOD__ || 'ngram_jaccard';
methodInput.value = defaultMethod;

const resultState = {
  payload: null,
  verdict: 'all',
  query: '',
  sort: 'desc',
  pageSize: 20,
  page: 1,
  openRows: new Set(),
  expandedFragments: new Set(),
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} МБ`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderSelectedFiles() {
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    selectedFiles.className = 'file-list empty';
    selectedFiles.textContent = 'Файлы ещё не выбраны.';
    return;
  }

  selectedFiles.className = 'file-list';
  selectedFiles.innerHTML = files.map((file) => `
    <div class="file-item">
      <div>
        <div class="file-name">${escapeHtml(file.name)}</div>
        <div class="file-meta">${escapeHtml(file.type || 'application/octet-stream')}</div>
      </div>
      <div class="file-meta">${formatBytes(file.size)}</div>
    </div>
  `).join('');
}

function renderSelectedGroundTruth() {
  const file = groundTruthInput.files?.[0];
  if (!file) {
    selectedGroundTruth.className = 'file-list empty';
    selectedGroundTruth.textContent = 'Эталонная разметка не выбрана. Сервис работает в обычном режиме.';
    return;
  }

  selectedGroundTruth.className = 'file-list';
  selectedGroundTruth.innerHTML = `
    <div class="file-item">
      <div>
        <div class="file-name">${escapeHtml(file.name)}</div>
        <div class="file-meta">Экспериментальный режим будет включён после запуска анализа</div>
      </div>
      <div class="file-meta">${formatBytes(file.size)}</div>
    </div>
  `;
}

function setActiveMethod(methodKey) {
  methodInput.value = methodKey;
  document.querySelectorAll('.method-card').forEach((button) => {
    button.classList.toggle('active', button.dataset.method === methodKey);
  });
}

function getDocTitle(docId) {
  const docs = resultState.payload?.uploaded_documents || [];
  const doc = docs.find((item) => item.doc_id === docId);
  return doc?.title || docId;
}

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '0.0000';
}

function formatMs(value) {
  return `${Number(value || 0).toFixed(2)} мс`;
}

function rowKey(row) {
  return `${row.left_id}__${row.right_id}`;
}

function compactText(value, limit = 420) {
  const text = String(value || '');
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}...`;
}

function getSummary(analysis, elapsedMs) {
  if (analysis.summary) return analysis.summary;
  const pairwise = analysis.pairwise || [];
  const scores = pairwise.map((row) => Number(row.score || 0));
  return {
    method: analysis.method,
    document_count: analysis.document_count,
    pair_count: pairwise.length,
    match_count: pairwise.filter((row) => row.verdict === 'match').length,
    no_match_count: pairwise.filter((row) => row.verdict === 'no_match').length,
    average_score: scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0,
    max_score: scores.length ? Math.max(...scores) : 0,
    min_score: scores.length ? Math.min(...scores) : 0,
    total_time_ms: elapsedMs || 0,
    average_time_per_pair_ms: pairwise.length ? (elapsedMs || 0) / pairwise.length : 0,
  };
}

function detailItem(label, value) {
  const normalized = Array.isArray(value) ? value.join(', ') : value;
  const displayValue = normalized === undefined || normalized === null || normalized === '' ? 'Нет данных' : normalized;
  return `
    <div class="detail-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(displayValue)}</strong>
    </div>
  `;
}

function renderFragment(row, key) {
  const metadata = row.metadata || {};
  const fragment = metadata.fragment || row.fragment || '';
  if (!fragment) return detailItem('Найденный фрагмент', 'Нет данных');

  const isLong = fragment.length > 420;
  const isExpanded = resultState.expandedFragments.has(key);
  const visibleText = isLong && !isExpanded ? compactText(fragment) : fragment;
  return `
    <div class="detail-fragment">
      <div class="detail-fragment-head">
        <span>Найденный общий фрагмент</span>
        ${isLong ? `<button type="button" class="link-button" data-action="toggle-fragment" data-key="${escapeHtml(key)}">${isExpanded ? 'Свернуть' : 'Развернуть'}</button>` : ''}
      </div>
      <pre>${escapeHtml(visibleText)}</pre>
    </div>
  `;
}

function renderMethodDetails(row) {
  const method = resultState.payload?.analysis?.method;
  const metadata = row.metadata || {};
  const key = rowKey(row);

  if (method === 'suffix_exact') {
    return `
      ${detailItem('Длина наибольшего общего фрагмента', metadata.longest_common_substring_length)}
      ${detailItem('Тип реализации', metadata.implementation)}
      ${renderFragment(row, key)}
    `;
  }

  if (method === 'minhash_lsh') {
    return `
      ${detailItem('Размер шингла', metadata.shingle_size)}
      ${detailItem('Реализация', metadata.implementation)}
      ${detailItem('Похожесть по сигнатурам', metadata.signature_jaccard ?? row.score)}
      ${detailItem('Кандидат LSH', metadata.candidate === undefined ? 'Нет данных' : (metadata.candidate ? 'Да' : 'Нет'))}
      ${detailItem('Число кандидатов', metadata.candidate_count)}
      ${detailItem('Общие шинглы', metadata.shared_shingle_count)}
      ${detailItem('Объединение шинглов', metadata.union_shingle_count)}
    `;
  }

  if (method === 'inverted_index') {
    return `
      ${detailItem('Число общих терминов', metadata.shared_terms_count)}
      ${detailItem('Размер объединения терминов', metadata.union_terms_count)}
      ${detailItem('Совпавшие ключевые слова', metadata.shared_terms)}
      ${detailItem('Формула score', metadata.score_formula)}
    `;
  }

  if (method === 'ngram_jaccard') {
    return `
      ${detailItem('Размер n-граммы', metadata.ngram_size)}
      ${detailItem('Общие n-граммы', metadata.shared_ngram_count)}
      ${detailItem('Объединение n-грамм', metadata.union_ngram_count)}
      ${detailItem('Коэффициент Жаккара', metadata.jaccard ?? row.score)}
      ${detailItem('Примеры общих n-грамм', metadata.shared_ngrams_preview)}
    `;
  }

  return `<pre>${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>`;
}

function renderGroundTruthDetails(row) {
  const metadata = row.metadata || {};
  if (!metadata.experiment_outcome) return '';
  return `
    ${detailItem('Ожидалось по эталону', metadata.expected_match ? 'match' : 'no_match')}
    ${detailItem('Экспериментальный исход', metadata.experiment_outcome)}
    ${detailItem('Сценарий', metadata.scenario)}
  `;
}

function renderDetailsRow(row) {
  return `
    <tr class="details-row">
      <td colspan="5">
        <div class="details-panel">
          <div class="details-grid">
            ${detailItem('Левый документ', `${row.left_id}: ${getDocTitle(row.left_id)}`)}
            ${detailItem('Правый документ', `${row.right_id}: ${getDocTitle(row.right_id)}`)}
            ${detailItem('Score', formatScore(row.score))}
            ${detailItem('Verdict', row.verdict)}
            ${renderGroundTruthDetails(row)}
            ${renderMethodDetails(row)}
          </div>
        </div>
      </td>
    </tr>
  `;
}

function filteredRows() {
  const rows = [...(resultState.payload?.analysis?.pairwise || [])];
  const query = resultState.query.trim().toLowerCase();
  const filtered = rows.filter((row) => {
    const verdictMatch = resultState.verdict === 'all' || row.verdict === resultState.verdict;
    const leftTitle = getDocTitle(row.left_id).toLowerCase();
    const rightTitle = getDocTitle(row.right_id).toLowerCase();
    const textMatch = !query || row.left_id.toLowerCase().includes(query) || row.right_id.toLowerCase().includes(query) || leftTitle.includes(query) || rightTitle.includes(query);
    return verdictMatch && textMatch;
  });

  filtered.sort((a, b) => {
    const diff = Number(a.score || 0) - Number(b.score || 0);
    return resultState.sort === 'asc' ? diff : -diff;
  });
  return filtered;
}

function renderPairwiseTable() {
  const rows = filteredRows();
  const totalPages = Math.max(Math.ceil(rows.length / resultState.pageSize), 1);
  resultState.page = Math.min(resultState.page, totalPages);
  const start = (resultState.page - 1) * resultState.pageSize;
  const pageRows = rows.slice(start, start + resultState.pageSize);

  if (!resultState.payload?.analysis?.pairwise?.length) {
    return '<div class="empty-state">Попарные результаты отсутствуют.</div>';
  }

  const body = pageRows.map((row) => {
    const key = rowKey(row);
    const isOpen = resultState.openRows.has(key);
    return `
      <tr>
        <td>
          <strong>${escapeHtml(row.left_id)}</strong>
          <span class="doc-title">${escapeHtml(getDocTitle(row.left_id))}</span>
        </td>
        <td>
          <strong>${escapeHtml(row.right_id)}</strong>
          <span class="doc-title">${escapeHtml(getDocTitle(row.right_id))}</span>
        </td>
        <td>${formatScore(row.score)}</td>
        <td><span class="pill ${row.verdict === 'match' ? 'match' : 'no-match'}">${escapeHtml(row.verdict)}</span></td>
        <td><button type="button" class="small-button" data-action="toggle-details" data-key="${escapeHtml(key)}">${isOpen ? 'Скрыть' : 'Подробнее'}</button></td>
      </tr>
      ${isOpen ? renderDetailsRow(row) : ''}
    `;
  }).join('');

  return `
    <div class="results-toolbar">
      <label>
        <span>Verdict</span>
        <select id="verdict-filter">
          <option value="all" ${resultState.verdict === 'all' ? 'selected' : ''}>Все</option>
          <option value="match" ${resultState.verdict === 'match' ? 'selected' : ''}>Только match</option>
          <option value="no_match" ${resultState.verdict === 'no_match' ? 'selected' : ''}>Только no_match</option>
        </select>
      </label>
      <label>
        <span>Поиск по документу</span>
        <input id="document-search" type="search" value="${escapeHtml(resultState.query)}" placeholder="Название или ID">
      </label>
      <label>
        <span>Сортировка Score</span>
        <select id="score-sort">
          <option value="desc" ${resultState.sort === 'desc' ? 'selected' : ''}>По убыванию</option>
          <option value="asc" ${resultState.sort === 'asc' ? 'selected' : ''}>По возрастанию</option>
        </select>
      </label>
      <label>
        <span>Строк на странице</span>
        <select id="page-size">
          <option value="20" ${resultState.pageSize === 20 ? 'selected' : ''}>20</option>
          <option value="50" ${resultState.pageSize === 50 ? 'selected' : ''}>50</option>
          <option value="100" ${resultState.pageSize === 100 ? 'selected' : ''}>100</option>
        </select>
      </label>
    </div>
    <div class="table-meta">Показано ${pageRows.length} из ${rows.length} результатов</div>
    <div class="table-shell compact-table">
      <table>
        <thead>
          <tr>
            <th>Левый документ</th>
            <th>Правый документ</th>
            <th>Score</th>
            <th>Verdict</th>
            <th>Детали</th>
          </tr>
        </thead>
        <tbody>${body || `<tr><td colspan="5" class="empty-cell">Нет результатов по выбранным фильтрам.</td></tr>`}</tbody>
      </table>
    </div>
    <div class="pagination-row">
      <button type="button" class="ghost-button" data-action="prev-page" ${resultState.page <= 1 ? 'disabled' : ''}>Назад</button>
      <span>Страница ${resultState.page} из ${totalPages}</span>
      <button type="button" class="ghost-button" data-action="next-page" ${resultState.page >= totalPages ? 'disabled' : ''}>Вперёд</button>
    </div>
  `;
}

function renderSummary(summary, methodMeta) {
  const cards = [
    ['Выбранный метод', methodMeta?.title || summary.method],
    ['Документов', summary.document_count],
    ['Попарных сравнений', summary.pair_count],
    ['Match', summary.match_count],
    ['No match', summary.no_match_count],
    ['Средний score', formatScore(summary.average_score)],
    ['Максимальный score', formatScore(summary.max_score)],
    ['Минимальный score', formatScore(summary.min_score)],
    ['Общее время', formatMs(summary.total_time_ms)],
    ['Среднее время на пару', formatMs(summary.average_time_per_pair_ms)],
  ];
  Object.entries(summary.method_specific_metrics || {}).forEach(([key, value]) => {
    cards.push([key, Number.isFinite(Number(value)) ? formatScore(value) : value]);
  });

  return `
    <div class="summary-grid wide-summary">
      ${cards.map(([label, value]) => `
        <div class="summary-card">
          <div class="summary-label">${escapeHtml(label)}</div>
          <div class="summary-value">${escapeHtml(value)}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderExperimentMetrics(metrics) {
  if (!metrics) return '';
  const cards = [
    ['TP', metrics.tp],
    ['FP', metrics.fp],
    ['FN', metrics.fn],
    ['TN', metrics.tn],
    ['Precision', formatScore(metrics.precision)],
    ['Recall', formatScore(metrics.recall)],
    ['F1', formatScore(metrics.f1)],
  ];

  return `
    <div class="experiment-block">
      <h3>Экспериментальные метрики</h3>
      <div class="experiment-grid">
        ${cards.map(([label, value]) => `
          <div class="experiment-card">
            <div class="experiment-label">${escapeHtml(label)}</div>
            <div class="experiment-value">${escapeHtml(value)}</div>
          </div>
        `).join('')}
      </div>
      <div class="table-meta">
        Оценено размеченных пар: ${escapeHtml(metrics.evaluated_pair_count)} из ${escapeHtml(metrics.labeled_pair_count)}.
        Неразмеченных результатов: ${escapeHtml(metrics.unlabeled_result_count || 0)}.
      </div>
    </div>
  `;
}

function renderSearchResults(analysis) {
  if (!analysis.search_results?.length) return '';
  return `
    <h3>Результаты поиска по дополнительному запросу</h3>
    <div class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Документ</th>
            <th>Score</th>
            <th>Метаданные</th>
          </tr>
        </thead>
        <tbody>
          ${analysis.search_results.map((row) => `
            <tr>
              <td>${escapeHtml(row.doc_id)}</td>
              <td>${formatScore(row.score)}</td>
              <td><pre>${escapeHtml(JSON.stringify(row.metadata || {}, null, 2))}</pre></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderDocMap(uploadedDocs) {
  return `
    <div class="doc-map">
      ${uploadedDocs.map((doc) => `<span class="doc-chip"><strong>${escapeHtml(doc.doc_id)}</strong> - ${escapeHtml(doc.title || doc.doc_id)}</span>`).join('')}
    </div>
  `;
}

function renderNotes(notes) {
  if (!notes?.length) return '';
  return `
    <ul class="notes-list">
      ${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('')}
    </ul>
  `;
}

function renderResults() {
  const payload = resultState.payload;
  if (!payload) return;
  const { analysis, uploaded_documents: uploadedDocs, elapsed_ms: elapsedMs, method_meta: methodMeta } = payload;
  const summary = getSummary(analysis, elapsedMs);

  resultsRoot.className = 'results-root';
  resultsRoot.innerHTML = `
    <div class="results-head">
      <h3>Сводка запуска</h3>
      <div class="export-actions">
        <button type="button" class="ghost-button" data-action="export-json">Сохранить JSON</button>
        <button type="button" class="ghost-button" data-action="export-txt">Сохранить TXT</button>
        <button type="button" class="ghost-button" data-action="export-csv">Сохранить CSV</button>
      </div>
    </div>
    ${renderSummary(summary, methodMeta)}
    ${renderExperimentMetrics(analysis.experiment_metrics)}
    ${renderDocMap(uploadedDocs || [])}
    <h3>Попарное сравнение</h3>
    ${renderPairwiseTable()}
    ${renderSearchResults(analysis)}
    ${renderNotes(analysis.notes)}
  `;
}

function buildExportData() {
  const payload = resultState.payload;
  const analysis = payload.analysis;
  return {
    exported_at: new Date().toISOString(),
    method: analysis.method,
    method_title: payload.method_meta?.title || analysis.method,
    parameters: analysis.parameters || {
      threshold: analysis.threshold,
    },
    summary: getSummary(analysis, payload.elapsed_ms),
    experiment_metrics: analysis.experiment_metrics || null,
    uploaded_documents: payload.uploaded_documents || [],
    pairwise_results: analysis.pairwise || [],
    search_results: analysis.search_results || [],
    notes: analysis.notes || [],
  };
}

function fileTimestamp() {
  const pad = (value) => String(value).padStart(2, '0');
  const now = new Date();
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
}

function downloadFile(content, type, extension) {
  const method = resultState.payload?.analysis?.method || 'analysis';
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${method}_results_${fileTimestamp()}.${extension}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportJson() {
  downloadFile(JSON.stringify(buildExportData(), null, 2), 'application/json;charset=utf-8', 'json');
}

function exportTxt() {
  const data = buildExportData();
  const lines = [
    `Дата и время запуска: ${data.exported_at}`,
    `Метод: ${data.method_title} (${data.method})`,
    `Параметры: ${JSON.stringify(data.parameters)}`,
    '',
    'Сводка:',
    ...Object.entries(data.summary).map(([key, value]) => `${key}: ${value}`),
    '',
    'Экспериментальные метрики:',
    ...(data.experiment_metrics ? Object.entries(data.experiment_metrics).map(([key, value]) => `${key}: ${value}`) : ['нет']),
    '',
    'Попарные результаты:',
    ...data.pairwise_results.map((row, index) => [
      `${index + 1}. ${row.left_id} - ${row.right_id}`,
      `score: ${row.score}`,
      `verdict: ${row.verdict}`,
      `fragment: ${row.fragment || ''}`,
      `metadata: ${JSON.stringify(row.metadata || {})}`,
    ].join('\n')),
  ];
  downloadFile(lines.join('\n'), 'text/plain;charset=utf-8', 'txt');
}

function csvCell(value) {
  return `"${String(value ?? '').replace(/"/g, '""')}"`;
}

function exportCsv() {
  const data = buildExportData();
  const metaRows = [
    ['section', 'key', 'value'],
    ['run', 'exported_at', data.exported_at],
    ['run', 'method', data.method],
    ['run', 'method_title', data.method_title],
    ['run', 'parameters', JSON.stringify(data.parameters)],
    ...Object.entries(data.summary).map(([key, value]) => ['summary', key, JSON.stringify(value)]),
    ...(data.experiment_metrics ? Object.entries(data.experiment_metrics).map(([key, value]) => ['experiment_metrics', key, value]) : []),
    [],
  ];
  const header = ['left_id', 'right_id', 'score', 'verdict', 'expected_match', 'experiment_outcome', 'scenario', 'fragment', 'metadata'];
  const rows = data.pairwise_results.map((row) => [
    row.left_id,
    row.right_id,
    row.score,
    row.verdict,
    row.metadata?.expected_match ?? '',
    row.metadata?.experiment_outcome ?? '',
    row.metadata?.scenario ?? '',
    row.fragment || '',
    JSON.stringify(row.metadata || {}),
  ]);
  downloadFile([...metaRows, header, ...rows].map((row) => row.map(csvCell).join(',')).join('\n'), 'text/csv;charset=utf-8', 'csv');
}

methodGrid.addEventListener('click', (event) => {
  const button = event.target.closest('.method-card');
  if (!button) return;
  setActiveMethod(button.dataset.method);
});

fileInput.addEventListener('change', renderSelectedFiles);
groundTruthInput.addEventListener('change', renderSelectedGroundTruth);

clearButton.addEventListener('click', () => {
  form.reset();
  methodInput.value = defaultMethod;
  setActiveMethod(defaultMethod);
  renderSelectedFiles();
  renderSelectedGroundTruth();
  resultState.payload = null;
  resultState.openRows.clear();
  resultState.expandedFragments.clear();
  resultsRoot.className = 'results-root empty-state';
  resultsRoot.textContent = 'Загрузите коллекцию и нажмите «Запустить анализ».';
  statusBanner.className = 'status-banner idle';
  statusBanner.textContent = 'Ожидание запуска анализа.';
});

resultsRoot.addEventListener('input', (event) => {
  if (event.target.id !== 'document-search') return;
  resultState.query = event.target.value;
  resultState.page = 1;
  renderResults();
});

resultsRoot.addEventListener('change', (event) => {
  if (event.target.id === 'verdict-filter') {
    resultState.verdict = event.target.value;
    resultState.page = 1;
    renderResults();
  }
  if (event.target.id === 'score-sort') {
    resultState.sort = event.target.value;
    renderResults();
  }
  if (event.target.id === 'page-size') {
    resultState.pageSize = Number(event.target.value);
    resultState.page = 1;
    renderResults();
  }
});

resultsRoot.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const action = button.dataset.action;
  const key = button.dataset.key;

  if (action === 'toggle-details') {
    if (resultState.openRows.has(key)) resultState.openRows.delete(key);
    else resultState.openRows.add(key);
    renderResults();
  }
  if (action === 'toggle-fragment') {
    if (resultState.expandedFragments.has(key)) resultState.expandedFragments.delete(key);
    else resultState.expandedFragments.add(key);
    renderResults();
  }
  if (action === 'prev-page' && resultState.page > 1) {
    resultState.page -= 1;
    renderResults();
  }
  if (action === 'next-page') {
    resultState.page += 1;
    renderResults();
  }
  if (action === 'export-json') exportJson();
  if (action === 'export-txt') exportTxt();
  if (action === 'export-csv') exportCsv();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (files.length < 2) {
    statusBanner.className = 'status-banner error';
    statusBanner.textContent = 'Для попарного анализа выберите минимум два документа.';
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  const groundTruthFile = groundTruthInput.files?.[0];
  if (groundTruthFile) formData.append('ground_truth_file', groundTruthFile);
  formData.append('method', methodInput.value);
  formData.append('threshold', document.getElementById('threshold').value);
  formData.append('shingle_size', document.getElementById('shingle_size').value);
  formData.append('ngram_size', document.getElementById('ngram_size').value);
  formData.append('top_k', document.getElementById('top_k').value);
  formData.append('query_text', document.getElementById('query_text').value);

  statusBanner.className = 'status-banner loading';
  statusBanner.textContent = 'Анализ выполняется...';
  resultsRoot.className = 'results-root empty-state';
  resultsRoot.textContent = 'Выполняется обработка документов.';

  try {
    const response = await fetch('/analyze-upload', {
      method: 'POST',
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Сервис вернул ошибку.');
    }

    resultState.payload = payload;
    resultState.verdict = 'all';
    resultState.query = '';
    resultState.sort = 'desc';
    resultState.pageSize = 20;
    resultState.page = 1;
    resultState.openRows.clear();
    resultState.expandedFragments.clear();
    renderResults();
    statusBanner.className = 'status-banner success';
    statusBanner.textContent = `Анализ завершён: метод ${payload.analysis.method}, время ${formatMs(payload.elapsed_ms)}.`;
  } catch (error) {
    statusBanner.className = 'status-banner error';
    statusBanner.textContent = error.message || 'Не удалось выполнить анализ.';
    resultsRoot.className = 'results-root empty-state';
    resultsRoot.textContent = 'При выполнении анализа возникла ошибка.';
  }
});

setActiveMethod(defaultMethod);
renderSelectedFiles();
renderSelectedGroundTruth();
