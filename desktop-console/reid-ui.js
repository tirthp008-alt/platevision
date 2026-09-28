/* Optional cross-camera matching. Candidate scores never change vehicle counts. */
window.PlateReID = (() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const enabled = () => $('reidentify').value === 'true';
  const base = () => $('api-url').value.replace(/\/$/, '');
  const score = value => Number.isFinite(value) ? value.toFixed(3) : '—';
  let busy = false;

  function setBusy(value) {
    busy = value;
    $('reidentify').disabled = busy || $('mode').value === 'browser';
    for (const id of ['camera-id','observed-at']) $(id).disabled = busy || !enabled() || $('mode').value === 'browser';
    for (const id of ['reid-refresh','route-save','reid-clear','route-from','route-to','route-min','route-max']) $(id).disabled = busy;
  }
  function appendForm(form) {
    if (!enabled()) return;
    const camera = $('camera-id').value.trim(), entered = $('observed-at').value;
    if (!camera || camera.length > 64) throw new Error('Cross-camera matching requires a camera ID of 1–64 characters.');
    if (!entered) throw new Error('Enter the actual photo time or recording start time in UTC for cross-camera matching.');
    // The field is explicitly labelled UTC; never infer capture time from upload time.
    const timestamp = new Date(`${entered}Z`);
    if (!Number.isFinite(timestamp.getTime())) throw new Error('Enter a valid observation time in UTC.');
    form.append('reidentify','true'); form.append('camera_id',camera); form.append('observed_at',timestamp.toISOString());
  }
  async function request(path, options) {
    const response = await fetch(base() + path, {...options, signal:AbortSignal.timeout(15000)});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Cross-camera service rejected the request.');
    return result;
  }
  function renderRoutes(result) {
    const routes = Array.isArray(result) ? result : result.routes || [];
    const rows = $('route-rows'); rows.replaceChildren();
    for (const route of routes) {
      const row = rows.insertRow(); row.insertCell().textContent = route.source_camera;
      row.insertCell().textContent = route.target_camera;
      row.insertCell().textContent = `${route.min_seconds}–${route.max_seconds} seconds`;
    }
    if (!routes.length) { const cell = rows.insertRow().insertCell(); cell.colSpan = 3; cell.textContent = 'No saved camera routes. Enter a route based on the actual camera locations.'; }
  }
  function plateEvidence(value) {
    if (typeof value === 'string') return value === 'not_available' ? 'No plate-text evidence' : value;
    if (!value) return 'No plate-text evidence';
    const matching = value.known_matching_characters || 0;
    const conflicting = value.known_conflicting_characters || 0;
    const labels = {
      none: 'No plate-text evidence',
      agreement: `${matching} characters agree${value.boost > 0 ? ` · +${score(value.boost)} score` : ''}`,
      low_confidence: 'OCR confidence too low · no score boost',
      full_plate_conflict: 'Conflicting full plate',
      partial_conflict: `${conflicting} characters conflict · review`,
      insufficient_known_characters: `${matching} matching characters · insufficient evidence`,
    };
    return labels[value.kind] || value.label || value.status || 'Plate comparison available';
  }
  function observationCell(row, camera, track, timestamp) {
    const cell = row.insertCell();
    cell.textContent = `${camera || '—'} · track ${track ?? '—'}`;
    if (Number.isFinite(timestamp)) {
      const date = new Date(timestamp * 1000);
      if (Number.isFinite(date.getTime())) {
        const time = document.createElement('small');
        time.className = 'reid-evidence';
        time.textContent = date.toISOString().replace('T',' ').replace(/\.\d{3}Z$/, ' UTC');
        cell.append(time);
      }
    }
    return cell;
  }
  function reviewStatus(observation, match, preferred) {
    if (observation.status === 'ambiguous') return observation.reason === 'source_already_assigned_in_batch'
      ? 'Ambiguous · candidate shared by another track' : 'Ambiguous · similar candidates need review';
    if (match) return preferred && (match.source_observation_id || match.observation_id) === preferred
      ? 'Preferred candidate · unverified' : 'Unverified candidate';
    if (observation.recorded === false) return 'Not recorded · observation is out of order';
    const reasons = observation.rejected_candidates || {};
    const rejected = [];
    if (reasons.no_route) rejected.push('no camera route');
    if (reasons.chronological_or_travel) rejected.push('capture order or travel time');
    if (reasons.below_appearance_threshold) rejected.push('appearance below threshold');
    if (reasons.conflicting_full_plate) rejected.push('conflicting plate text');
    return rejected.length ? `Unmatched · ${rejected.join('; ')}` : 'Unmatched · saved for later cameras';
  }
  function renderResult(result) {
    if (!result) return;
    const rows = $('reid-rows'); rows.replaceChildren();
    const observations = result.observations || [];
    $('reid-latency').textContent = Number.isFinite(result.processing_time_ms) ? `${result.processing_time_ms.toFixed(1)} ms` : '—';
    $('reid-result-note').textContent = `${result.model || 'Appearance model'} · ${result.warning || result.note || 'Candidate matches are unverified. Appearance scores are not identity probabilities; vehicle counts remain unchanged.'}`;
    let count = 0;
    for (const observation of observations) {
      const candidates = observation.matches || observation.candidates || [];
      const matches = candidates.length ? candidates.slice(0,3) : [null];
      const preferred = typeof observation.selected_candidate === 'string' ? observation.selected_candidate
        : observation.selected_candidate?.source_observation_id || observation.selected_candidate?.observation_id;
      for (const match of matches) {
        const row = rows.insertRow(); count++;
        observationCell(row, observation.camera_id, observation.track_id, observation.first_seen_at ?? observation.observed_at);
        if (match) observationCell(row, match.source_camera || match.camera_id, match.source_track_id ?? match.track_id, match.source_last_seen_at ?? match.source_observed_at);
        else row.insertCell().textContent = 'No candidate';
        row.insertCell().textContent = reviewStatus(observation, match, preferred);
        row.insertCell().textContent = score(match?.appearance_similarity);
        row.insertCell().textContent = score(match?.score ?? match?.review_score);
        const travel = row.insertCell();
        travel.textContent = Number.isFinite(match?.travel_seconds) ? `${match.travel_seconds.toFixed(1)} s` : '—';
        if (Number.isFinite(match?.min_seconds) && Number.isFinite(match?.max_seconds)) {
          const allowed = document.createElement('small'); allowed.className = 'reid-evidence';
          allowed.textContent = `Allowed ${match.min_seconds}–${match.max_seconds} s`; travel.append(allowed);
        }
        row.insertCell().textContent = plateEvidence(match?.plate_evidence);
      }
    }
    if (!count) { const cell = rows.insertRow().insertCell(); cell.colSpan = 7; cell.className = 'empty-row'; cell.textContent = result.status === 'disabled' ? 'Cross-camera matching is off.' : ['error','failed','unavailable'].includes(result.status) ? 'Cross-camera matching unavailable. Plate and vehicle results remain available.' : 'No cross-camera candidates available.'; }
  }
  async function refresh() {
    $('reid-service').textContent = 'Checking cross-camera service…';
    try {
      const health = await request('/api/reid/health');
      $('reid-service').textContent = `${health.ready ? 'Ready' : 'Not ready'} · ${health.model || 'Appearance model'}. ${health.note || ''}`;
      const routes = await request('/api/reid/routes'); renderRoutes(routes);
      const matches = await request('/api/reid/matches'); renderResult(matches.reidentification || matches);
    } catch (error) { $('reid-service').textContent = error.message; }
  }
  async function saveRoute() {
    const from = $('route-from').value.trim(), to = $('route-to').value.trim();
    const lower = $('route-min').value, upper = $('route-max').value;
    const min = Number(lower), max = Number(upper);
    if (!from || !to || from.length > 64 || to.length > 64 || from === to || lower === '' || upper === '' || !Number.isFinite(min) || !Number.isFinite(max) || min < 0 || max < min) {
      $('route-status').textContent = 'Enter two different camera IDs and a valid minimum/maximum travel time in seconds.'; return;
    }
    $('route-save').disabled = true;
    try {
      const current = await request('/api/reid/routes');
      const routes = [...(Array.isArray(current) ? current : current.routes || [])];
      const route = {source_camera:from,target_camera:to,min_seconds:min,max_seconds:max};
      const index = routes.findIndex(item => item.source_camera === from && item.target_camera === to);
      if (index < 0) routes.push(route); else routes[index] = route;
      const saved = await request('/api/reid/routes',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({routes})});
      renderRoutes(saved);
      renderResult({observations:[],note:'Camera routes changed. Analyse the destination image or recording again to apply the saved travel times.'});
      $('route-status').textContent = `Saved ${from} → ${to}: ${min}–${max} seconds. Other routes retained. Reverse travel needs a separate route.`;
    } catch (error) { $('route-status').textContent = error.message; }
    finally { $('route-save').disabled = busy; }
  }
  async function clearGallery() {
    $('reid-clear').disabled = true;
    try { await request('/api/reid/gallery',{method:'DELETE'}); renderResult({observations:[],status:'empty'}); $('reid-service').textContent = 'Matching gallery cleared. Camera routes are unchanged.'; }
    catch (error) { $('reid-service').textContent = error.message; }
    finally { $('reid-clear').disabled = busy; }
  }
  $('reidentify').onchange = () => { setBusy(busy); if (enabled()) void refresh(); };
  $('reid-refresh').onclick = () => void refresh();
  $('route-save').onclick = () => void saveRoute();
  $('reid-clear').onclick = () => void clearGallery();
  setBusy(false);
  return {enabled,appendForm,setBusy,renderResult,refresh};
})();
