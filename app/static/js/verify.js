// ── Disagree toggle ───────────────────────────────────────────────────────────
function toggleDisagree() {
  const f = document.getElementById('disagree-form');
  f.style.display = (f.style.display === 'none' || f.style.display === '') ? 'block' : 'none';
}

// ── Submit ────────────────────────────────────────────────────────────────────
function submitAction(action) {
  document.getElementById('action-input').value = action;
  if (action === 'disagree') {
    document.getElementById('tv-input').value    = document.getElementById('tv-select').value;
    document.getElementById('fs-input').value    = document.getElementById('fs-select').value;
    document.getElementById('notes-input').value = document.getElementById('notes-area').value;
  }
  document.getElementById('main-form').submit();
}

// ── Event delegation (replaces all inline onclick handlers) ───────────────────
document.addEventListener('click', function(e) {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const action = el.dataset.action;
  if      (action === 'toggle-disagree')  toggleDisagree();
  else if (action === 'submit')           submitAction(el.dataset.value);
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e) {
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.key === 'y' || e.key === 'Y') submitAction('agree');
  if (e.key === 's' || e.key === 'S') submitAction('skip');
  if (e.key === 'n' || e.key === 'N') toggleDisagree();
});
