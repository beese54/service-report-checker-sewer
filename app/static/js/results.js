// ── Search / filter ───────────────────────────────────────────────────────────
document.getElementById('searchInput').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  document.querySelectorAll('#tableBody tr:not(.detail-row)').forEach(tr => {
    const name   = tr.dataset.name   || '';
    const status = tr.dataset.status || '';
    const type   = tr.dataset.type   || '';
    const match  = (name + status + type).toLowerCase().includes(q);
    tr.classList.toggle('d-none', !match);
    const next = tr.nextElementSibling;
    if (next && next.classList.contains('detail-row')) {
      next.classList.add('d-none');
    }
  });
});

// ── Sort ──────────────────────────────────────────────────────────────────────
let _sortCol = -1, _sortAsc = true;
document.querySelectorAll('#folderTable thead th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = parseInt(th.dataset.col);
    if (_sortCol === col) _sortAsc = !_sortAsc;
    else { _sortCol = col; _sortAsc = true; }
    sortTable(col, _sortAsc);
  });
});

function sortTable(col, asc) {
  const tbody = document.getElementById('tableBody');
  const rows  = Array.from(tbody.querySelectorAll('tr:not(.detail-row)'));
  rows.sort((a, b) => {
    const av = a.cells[col] ? a.cells[col].textContent.trim() : '';
    const bv = b.cells[col] ? b.cells[col].textContent.trim() : '';
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(r => {
    tbody.appendChild(r);
    const next = r.nextElementSibling;
    if (next && next.classList.contains('detail-row')) tbody.appendChild(next);
  });
}

// ── Detail toggle (detail HTML is pre-rendered by Jinja2, just toggle visibility)
document.getElementById('tableBody').addEventListener('click', function(e) {
  const btn = e.target.closest('[data-action="toggle-detail"]');
  if (!btn) return;
  const detailRow = btn.closest('tr').nextElementSibling;
  if (!detailRow || !detailRow.classList.contains('detail-row')) return;
  const isHidden = detailRow.classList.contains('d-none');
  detailRow.classList.toggle('d-none', !isHidden);
  btn.textContent = isHidden ? 'Hide' : 'Show';
});
