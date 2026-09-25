/* Native recording workflow: upload once, then play/download the annotated MP4. */
window.PlateExport = (() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let job = null, file = null, epoch = 0, timer = null, phase = 'idle', base = '';
  const active = () => file !== null;
  function controls() {
    $('detect-button').disabled = ['uploading', 'queued', 'processing'].includes(phase);
    $('detect-button').textContent = phase === 'paused' ? 'Resume video processing' : 'Process video again';
    $('stop-button').disabled = !['processing', 'queued'].includes(phase);
    $('stop-button').textContent = 'Pause';
    $('video-profile').disabled = ['uploading', 'queued', 'processing', 'paused'].includes(phase);
    $('video-interval').disabled = true;
  }
  function reset() {
    const previous = job, previousBase = base;
    epoch++; clearTimeout(timer); file = null; job = null; phase = 'idle';
    if (previous && !['complete', 'failed', 'cancelled'].includes(previous.status)) {
      fetch(`${previousBase}/api/videos/${previous.id}/cancel`, {method:'POST'}).catch(() => {});
    }
    const player = $('annotated-video'); player.pause(); player.removeAttribute('src'); player.load(); player.hidden = true;
    $('preview').hidden = false; $('video-export-actions').hidden = true;
    $('video-profile').disabled = false; $('latency-title').textContent = 'Frame latency'; $('plate-count-note').textContent = 'Current analysed frame'; $('ocr-note').textContent = 'Scores are model estimates, not verified accuracy';
  }
  async function json(response) {
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Recording request failed.');
    return result;
  }
  function render(result) {
    job = result; phase = result.status; controls(); $('latency-title').textContent = 'Processing / frame'; $('plate-count-note').textContent = 'Tracks across the recording';
    $('recording-progress').hidden = false;
    $('recording-meter').value = result.progress || 0;
    const phaseLabel = phase === 'paused' ? 'Paused' : result.phase;
    $('recording-position').textContent = `${Math.round(result.progress || 0)}% · ${phaseLabel}`;
    $('status').textContent = result.status === 'failed' ? result.error : `${phaseLabel}. ${result.frames_processed || 0} of ${result.total_frames || result.frames || '…'} frames.`;
    if (result.frame_ms) {
      $('latency').textContent = `${Math.round(result.frame_ms)} ms`;
      $('latency-note').textContent = 'Decode + full-scene detection + tracking · OCR/export pending';
    }
    if (result.status !== 'complete') return;
    const player = $('annotated-video');
    player.src = base + result.video_url; player.hidden = false; player.load();
    $('preview').hidden = true; $('empty').hidden = true; $('video').hidden = true;
    $('video-export-actions').hidden = false;
    $('download-video').href = base + result.video_url + '?download=true';
    $('download-report').href = base + result.report_url;
    $('latency').textContent = `${result.processing_ms_per_frame.toFixed(1)} ms`;
    $('latency-note').textContent = `Overall average/frame, including OCR + MP4 export · detector p95 ${result.detection_ms.p95} ms`;
    $('ocr-latency').textContent = `${Math.round(result.ocr_total_ms)} ms`;
    $('ocr-note').textContent = 'Total OCR across the complete recording';
    $('plate-count').textContent = result.tracks.length;
    $('vehicle-count').textContent = result.tracks.filter(t => t.text).length;
    $('coverage').textContent = `${result.frames} frames searched · ${result.tracks.length} plate tracks`;
    $('status').textContent = `Annotated video ready. Every frame was searched with ${result.engine}. ${result.processing_ms_per_frame < 50 ? 'Overall processing averaged under 50 ms/frame.' : 'Overall processing exceeded the 50 ms/frame target.'} Play or download below. Readings use the clearest observed crop in each track; verify uncertain text.`;
    $('recording-log').hidden = false;
    const rows = $('recording-rows'); rows.replaceChildren();
    const cards = $('vehicle-cards'); cards.replaceChildren();
    for (const track of result.tracks) {
      const row = rows.insertRow();
      row.insertCell().textContent = `${track.first_seen.toFixed(2)}–${track.last_seen.toFixed(2)} s`;
      const image = document.createElement('img'); image.src = base + track.crop_url; image.alt = `Plate track ${track.id}`;
      row.insertCell().append(image); row.insertCell().textContent = track.text || 'Unreadable';
      row.insertCell().textContent = track.observations;
      row.insertCell().textContent = track.format_status === 'valid' ? `${Math.round(track.ocr_confidence*100)}% · verify crop` : 'Uncertain · review crop';
      const card = document.createElement('article'); card.className = 'vehicle-card';
      const copy = document.createElement('div'), title = document.createElement('span'), text = document.createElement('strong'), detail = document.createElement('small');
      title.className = 'vehicle-title'; title.textContent = `Plate track ${track.id}`; text.textContent = track.text || 'Unreadable';
      detail.textContent = `${track.observations} frames · ${Math.round(track.ocr_confidence*100)}% OCR confidence`;
      copy.append(title,text,detail); card.append(image.cloneNode(),copy); cards.append(card);
    }
    if (!result.tracks.length) {
      const cell = rows.insertRow().insertCell(); cell.colSpan = 5; cell.textContent = 'No confirmed plate regions in this recording.';
      cards.textContent = 'No plate regions found. Try the detailed profile for small or distant plates.';
    }
    const currentRows = $('result-rows'); currentRows.replaceChildren();
    const cell = currentRows.insertRow().insertCell(); cell.colSpan = 6; cell.className = 'empty-row';
    cell.textContent = 'See the recording detections below for all plate tracks.';
  }
  async function poll(token) {
    if (token !== epoch || !job) return;
    try {
      const result = await json(await fetch(`${base}/api/videos/${job.id}`, {signal:AbortSignal.timeout(10000)}));
      if (token !== epoch) return;
      render(result);
      if (['queued', 'processing', 'paused'].includes(phase)) timer = setTimeout(() => poll(token), 500);
    } catch (error) {
      if (token !== epoch) return;
      $('status').textContent = `Waiting for video service: ${error.message}`;
      timer = setTimeout(() => poll(token), 1500);
    }
  }
  async function start(selectedFile) {
    reset(); file = selectedFile; const token = epoch;
    base = $('api-url').value.replace(/\/$/, ''); phase = 'uploading'; controls();
    $('recording-progress').hidden = false; $('recording-meter').value = 0;
    $('recording-position').textContent = 'Uploading video once…';
    $('status').textContent = 'Uploading recording for every-frame plate detection and annotated MP4 export.';
    const form = new FormData(); form.append('video', selectedFile); form.append('profile', $('video-profile').value);
    try {
      const result = await json(await fetch(`${base}/api/videos`, {method:'POST', body:form}));
      if (token !== epoch) { fetch(`${base}/api/videos/${result.id}/cancel`, {method:'POST'}).catch(() => {}); return; }
      render(result); void poll(token);
    } catch (error) {
      if (token !== epoch) return;
      phase = 'failed'; controls(); $('status').textContent = `Video processing failed: ${error.message}`;
    }
  }
  async function action(name) {
    if (!job) return;
    const token = epoch;
    try {
      const result = await json(await fetch(`${base}/api/videos/${job.id}/${name}`, {method:'POST'}));
      if (token === epoch) render(result);
    } catch (error) { if (token === epoch) $('status').textContent = error.message; }
  }
  return {active, controls, reset, start, pause:() => action('pause'),
    resume:() => phase === 'paused' ? action('resume') : start(file)};
})();
