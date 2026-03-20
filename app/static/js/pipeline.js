// ── Drop-zone wiring ──────────────────────────────────────────────────────────
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('pipelineFile');
const fileInfo   = document.getElementById('fileInfo');
const runBtn     = document.getElementById('runBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(f) {
  fileInfo.textContent = f.name + ' (' + (f.size / 1024 / 1024).toFixed(1) + ' MB)';
  fileInfo.classList.remove('d-none');
  dropZone.querySelector('#dropLabel').textContent = 'Selected: ' + f.name;
  runBtn.disabled = false;
  runBtn._file = f;
}

// ── Run button ────────────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  const f = runBtn._file;
  if (!f) return;

  runBtn.disabled = true;
  document.getElementById('uploadSpinner').classList.remove('d-none');

  const fd = new FormData();
  fd.append('file', f);

  let jobId;
  try {
    const resp = await fetch('/run', { method: 'POST', body: fd });
    if (!resp.ok) {
      let msg = resp.statusText;
      try { const err = await resp.json(); msg = err.error || msg; }
      catch (_) { msg = (await resp.text().catch(() => msg)); }
      alert('Upload failed (' + resp.status + '): ' + msg);
      runBtn.disabled = false;
      document.getElementById('uploadSpinner').classList.add('d-none');
      return;
    }
    const data = await resp.json();
    jobId = data.job_id;
  } catch (e) {
    alert('Upload error: ' + e);
    runBtn.disabled = false;
    document.getElementById('uploadSpinner').classList.add('d-none');
    return;
  }

  document.getElementById('uploadSpinner').classList.add('d-none');
  document.getElementById('progressSection').classList.remove('d-none');

  // ── SSE progress ────────────────────────────────────────────────────────────
  const progressBar   = document.getElementById('progressBar');
  const progressLabel = document.getElementById('progressLabel');
  const progressCount = document.getElementById('progressCount');
  const logBox        = document.getElementById('logBox');

  let total = 0;
  let done  = 0;

  const evtSource = new EventSource('/stream/' + jobId);

  evtSource.addEventListener('init', e => {
    const parts = parseKV(e.data);
    total = parseInt(parts.total) || 0;
    progressLabel.textContent = 'Processing ' + total + ' folders\u2026';
  });

  evtSource.addEventListener('progress', e => {
    const parts = parseKV(e.data);
    done  = parseInt(parts.done)  || done;
    total = parseInt(parts.total) || total;
    const folder = parts.folder || '';
    const status = parts.status || '';
    const pct    = total > 0 ? Math.round(done / total * 100) : 0;

    progressBar.style.width = pct + '%';
    progressCount.textContent = done + ' / ' + total;
    appendLog(folder + '  ' + status, statusColour(status));
  });

  evtSource.addEventListener('status', e => {
    appendLog(e.data, '#88aaff');
  });

  evtSource.addEventListener('log', e => {
    appendLog(e.data, '#ccc');
  });

  evtSource.addEventListener('done', e => {
    evtSource.close();
    progressBar.classList.remove('progress-bar-animated');
    progressBar.style.width = '100%';
    progressLabel.textContent = 'Done!';
    appendLog('Pipeline complete. Redirecting to results\u2026', '#80ff80');
    setTimeout(() => { window.location.href = '/results/' + jobId; }, 800);
  });

  evtSource.addEventListener('error', e => {
    evtSource.close();
    if (e.data) {
      // Server sent a real ERROR event — pipeline failed
      appendLog('ERROR: ' + e.data, '#ff6666');
      setTimeout(() => { window.location.href = '/results/' + jobId; }, 2000);
    } else {
      // Connection dropped (proxy timeout) — pipeline still running, poll for completion
      appendLog('Progress stream disconnected (proxy timeout) — polling for job completion…', '#ffd080');
      pollUntilDone(jobId);
    }
  });
});

function pollUntilDone(jobId) {
  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/status/' + jobId);
      if (!resp.ok) return; // server temporarily unreachable, keep polling
      // Guard: if the server restarted and session expired, /status is now
      // public — but if the content-type is HTML we got a login redirect for
      // a different endpoint, so bail gracefully.
      const ct = resp.headers.get('content-type') || '';
      if (!ct.includes('application/json')) {
        clearInterval(interval);
        appendLog('Session expired — please log in again.', '#ffd080');
        setTimeout(() => { window.location.href = '/login'; }, 1500);
        return;
      }
      const data = await resp.json();
      if (data.status === 'done') {
        clearInterval(interval);
        appendLog('Pipeline complete. Redirecting to results…', '#80ff80');
        setTimeout(() => { window.location.href = '/results/' + jobId; }, 800);
      } else if (data.status === 'error') {
        clearInterval(interval);
        appendLog('Pipeline failed: ' + (data.error || 'unknown error'), '#ff6666');
        setTimeout(() => { window.location.href = '/results/' + jobId; }, 2000);
      }
      // status === 'running' → keep polling
    } catch (_) {
      // Network hiccup — keep polling
    }
  }, 5000);
}

function appendLog(msg, colour) {
  const logBox = document.getElementById('logBox');
  const line   = document.createElement('div');
  line.style.color  = colour || '#d4d4d4';
  line.textContent  = msg;
  logBox.appendChild(line);
  logBox.scrollTop  = logBox.scrollHeight;
}

function parseKV(str) {
  const out = {};
  str.split(/\s+/).forEach(part => {
    const idx = part.indexOf('=');
    if (idx > 0) out[part.slice(0, idx)] = part.slice(idx + 1);
  });
  return out;
}

function statusColour(s) {
  if (s === 'ACCEPTED')              return '#80ff80';
  if (s === 'NEEDS_REVIEW')          return '#ffd080';
  if (s === 'REJECTED')              return '#ff6666';
  if (s === 'OBSTRUCTION_PROCESSED') return '#80ccff';
  return '#d4d4d4';
}
