/* Native recording workflow: upload once, then play/download the annotated MP4. */
window.PlateExport = (() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let job = null, file = null, epoch = 0, timer = null, phase = 'idle', base = '';
  const active = () => file !== null;
  const wantsText = () => $('read-text').value === 'true';
  const categories = {car:'Cars', two_wheeler:'Two-wheelers', bus:'Buses', truck:'Trucks', bicycle:'Bicycles',
    three_wheeler:'Autos / three-wheelers', lcv:'Light commercial / cargo tempo', tempo:'Tempo Travellers', van:'Vans', other:'Other'};
  const baseCategories = ['car','two_wheeler','bus','truck','bicycle','other'];
  const categoryLabel = category => ({car:'Car', two_wheeler:'Two-wheeler', bus:'Bus', truck:'Truck', bicycle:'Bicycle',
    three_wheeler:'Auto / three-wheeler', auto_rickshaw:'Auto-rickshaw', lcv:'Light commercial / cargo tempo', tempo:'Tempo Traveller', van:'Van', other:'Other'})[category] || 'Other';
  function vehiclePlateSearchTiming(search) {
    if (!search || !(search.searched_vehicles > 0 || search.added_regions > 0)) return '';
    const recording = search.scope === 'recording', added = search.added_regions ?? 0;
    return ` Vehicle crop search: ${search.searched_vehicles ?? 0} ${recording ? 'per-frame ' : ''}checks · ${added} additional plate ${recording ? 'observation' : 'region'}${added === 1 ? '' : 's'}${Number.isFinite(search.time_ms) ? ` · ${search.time_ms.toFixed(1)} ms` : ''}.`;
  }
  function renderAnalysisSummary(summary, {plateCount, vehicleSummary, scope = 'frame', readText = true, pending = false} = {}) {
    const recording = scope === 'recording', recognized = readText && summary?.enabled === true;
    const vehicleAvailable = vehicleSummary?.available === true && !vehicleSummary.incomplete;
    const shown = value => Number.isFinite(value) ? value : '—';
    const values = [
      [recording ? 'Plate region tracks' : 'Plate regions', plateCount],
      ['Trusted distinct registrations', recognized ? summary.unique_registrations : null],
      ['Unresolved readings', recognized ? summary.unresolved_track_count : null],
      ['Ambiguous readings', recognized ? summary.ambiguous_track_count : null],
      [recording ? 'Vehicle tracks' : 'Vehicles detected', vehicleAvailable ? vehicleSummary.total_vehicles : null],
      [recording ? 'Repeat tracks grouped' : 'Repeat regions grouped', recognized ? summary.suppressed_repeat_count : null],
    ];
    $('analysis-summary-scope').textContent = recording ? 'WHOLE RECORDING' : 'CURRENT FRAME';
    const panel = $('recognition-breakdown'); panel.replaceChildren();
    for (const [label, value] of values) {
      const card = document.createElement('div'); card.className = 'category-count';
      const title = document.createElement('span'); title.textContent = label;
      const count = document.createElement('strong'); count.textContent = shown(value);
      card.append(title,count); panel.append(card);
    }
    const state = pending ? 'Recording analysis is in progress. ' : !readText ? 'OCR is off; registration counts are unavailable. ' : !recognized ? 'Registration summary is unavailable for this result. ' : '';
    const categories = vehicleSummary?.classification_review_count;
    $('recognition-summary-note').textContent = `${state}${recording ? 'Plate and vehicle totals count geometric tracks and may include fragments. ' : 'Plate regions and vehicles are separate visible detections. '}Trusted registrations meet the OCR checks; they are not verified identities. Accuracy requires labelled plate and vehicle examples.${Number.isFinite(categories) && categories > 0 ? ` ${categories} vehicle categor${categories === 1 ? 'y needs' : 'ies need'} review.` : ''}`;
  }
  function renderVehicleSummary(summary, vehicles = []) {
    const incomplete = summary?.incomplete === true;
    const available = summary?.available === true && !incomplete;
    const recording = summary?.scope === 'recording';
    $('vehicle-count').textContent = available ? (summary.total_vehicles ?? vehicles.length) : '—';
    $('vehicle-note').textContent = incomplete ? 'Vehicle analysis incomplete · totals unavailable' : !available ? 'Vehicle detection unavailable' : recording ? 'Vehicle tracks across recording · may fragment' : 'Visible vehicle detections in current frame';
    $('vehicle-scope').textContent = incomplete ? 'INCOMPLETE' : recording ? 'RECORDING TRACKS' : 'CURRENT FRAME';
    const panel = $('vehicle-breakdown'); panel.replaceChildren();
    const supported = summary?.supported_categories?.length ? summary.supported_categories : baseCategories;
    for (const key of supported) {
      const label = categories[key] || categoryLabel(key);
      const card = document.createElement('div'); card.className = 'category-count';
      const title = document.createElement('span'); title.textContent = label;
      const count = document.createElement('strong'); count.textContent = available ? (summary.counts_by_category?.[key] ?? 0) : '—';
      card.append(title, count); panel.append(card);
    }
    const unsupported = summary?.unsupported_categories || [];
    $('vehicle-capability-note').textContent = !available ? 'Category support depends on the loaded vehicle model.'
      : unsupported.length ? `Not classified separately by this model: ${unsupported.map(categoryLabel).join(', ')}. These may receive another supported label.`
      : supported.includes('tempo') && supported.includes('lcv') ? 'Categories follow the loaded vehicle model. Tempo Traveller and light commercial / cargo tempo are separate classes.'
      : 'Category support follows the loaded vehicle model.';
    const summaryText = !available ? 'Vehicle detection is unavailable. Plate regions are still searched independently.'
      : recording ? `${summary.visible_now ?? 0} visible in the last frame · ${summary.max_visible ?? 0} maximum simultaneously · ${summary.confirmed_tracks ?? 0} confirmed tracks · ${summary.tentative_tracks ?? 0} tentative. ${summary.counting_note || 'Geometric tracking can split one vehicle into multiple tracks after occlusion.'}`
      : `${summary.total_vehicles ?? vehicles.length} visible vehicle detections. ${summary.classification_note || 'Plate regions are detected independently of vehicles.'}`;
    $('vehicle-summary-note').textContent = `${incomplete ? 'Incomplete vehicle analysis. ' : ''}${summary?.warning ? summary.warning + ' ' : ''}${summaryText}`;
    const rows = $('vehicle-rows'); rows.replaceChildren();
    for (const vehicle of vehicles) {
      const row = rows.insertRow(); row.insertCell().textContent = `Vehicle ${String(vehicle.id ?? '').replace(/^vehicle[-_]/i, '')}`;
      const categoryCell = row.insertCell(); categoryCell.textContent = categoryLabel(vehicle.category);
      if (vehicle.classification_review?.requires_review) {
        const note = document.createElement('small'); note.className = 'reid-evidence';
        const candidates = (vehicle.classification_review.candidates || []).slice(0,3);
        note.textContent = `Category needs review${candidates.length ? ': ' + candidates.map(candidate => `${categoryLabel(candidate.category)} (${candidate.observations ?? 0} observations)`).join('; ') : ''}`;
        categoryCell.append(note);
      }
      const confidence = vehicle.classification_confidence ?? vehicle.confidence ?? vehicle.detection_confidence;
      row.insertCell().textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : '—';
      row.insertCell().textContent = recording ? `${vehicle.observations ?? 0} frame observations · ${vehicle.confirmed ? 'confirmed' : 'tentative'} track`
        : vehicle.plate_ids?.length ? vehicle.plate_ids.join(', ') : 'No plate associated';
    }
    if (!vehicles.length) { const cell = rows.insertRow().insertCell(); cell.colSpan = 4; cell.className = 'empty-row'; cell.textContent = available ? 'No vehicles detected.' : 'Vehicle detection unavailable.'; }
  }
  function appendRectification(parent, rectification, cropUrl, apiBase) {
    if (!rectification?.applied || typeof cropUrl !== 'string' || !cropUrl.startsWith('/api/')) return;
    const details = document.createElement('details'); details.className = 'rectified-evidence';
    const title = document.createElement('summary'); title.textContent = 'Angle-corrected view';
    const image = document.createElement('img'); image.src = apiBase + cropUrl; image.alt = 'Angle-corrected plate crop; original retained alongside'; image.loading = 'lazy';
    const note = document.createElement('small'); note.textContent = `${Number.isFinite(rectification.angle_degrees) ? `${rectification.angle_degrees.toFixed(1)}° estimated tilt · ` : ''}Separate corrected evidence. A corrected view does not verify plate text.`;
    details.append(title, image, note); parent.append(details);
  }
  function appendGreenEnhancement(parent, enhancement, apiBase, showPreview = true) {
    if (!enhancement?.applied) return;
    const details = document.createElement('details'); details.className = 'hint';
    const heading = document.createElement('summary');
    heading.textContent = 'Emboss + gradient · review required';
    const reading = document.createElement('p');
    const score = Number.isFinite(enhancement.candidate_ocr_confidence)
      ? `${Math.round(enhancement.candidate_ocr_confidence * 100)}% OCR score` : 'No OCR score';
    reading.textContent = `Experimental alternative: ${enhancement.candidate_text || 'Unreadable'} · ${score}. Original reading retained.`;
    details.append(heading, reading);
    if (showPreview && typeof enhancement.crop_url === 'string' && enhancement.crop_url.startsWith('/api/')) {
      const image = document.createElement('img'); image.src = apiBase + enhancement.crop_url;
      image.alt = 'Experimental embossed plate crop; compare with original'; image.loading = 'lazy';
      details.append(image);
    }
    const note = document.createElement('small');
    const extra = (Number(enhancement.preprocessing_ms) || 0) + (Number(enhancement.ocr_ms) || 0);
    note.textContent = `Added processing: ${extra.toFixed(1)} ms. Uncalibrated scores; verify characters against the original crop.`;
    details.append(note); parent.append(details);
  }
  function appendOcrReview(parent, review) {
    if (!review) return;
    const attempts = Array.isArray(review.attempts) ? review.attempts : [];
    const conflicts = Array.isArray(review.conflicting_readings) ? review.conflicting_readings.length > 0 : Boolean(review.conflicting_readings);
    if (attempts.length < 2 && !review.requires_review && !conflicts) return;
    const details = document.createElement('details'); details.className = 'hint';
    const heading = document.createElement('summary');
    heading.textContent = `OCR cleanup · ${attempts.length} attempt${attempts.length === 1 ? '' : 's'}${review.requires_review || conflicts ? ' · review required' : ''}`;
    details.append(heading);
    const selected = document.createElement('p');
    selected.textContent = `Selected view: ${(review.selected_method || 'original').replace(/_/g,' ')}.${conflicts ? ' Readings disagree; compare characters with the original crop.' : ''}`;
    details.append(selected);
    for (const attempt of attempts) {
      const item = document.createElement('p');
      item.textContent = `${String(attempt.method || 'original').replace(/_/g,' ')}: ${attempt.text || 'Unreadable'} · ${Number.isFinite(attempt.confidence) ? `${Math.round(attempt.confidence * 100)}% OCR score` : 'No OCR score'}`;
      details.append(item);
    }
    parent.append(details);
  }
  function recordingReadings(result, readText) {
    const tracks = result.tracks || [], summary = result.recognized_plate_summary;
    if (!readText || !summary?.enabled || !Array.isArray(summary.groups)) return tracks;
    const byId = new Map(tracks.map(track => [String(track.id), track])), displayed = new Set(), readings = [];
    for (const group of summary.groups) {
      const representative = byId.get(String(group.representative_track_id));
      if (!representative || !Array.isArray(group.track_ids) || !group.track_ids.length) continue;
      const members = group.track_ids.map(id => byId.get(String(id))).filter(Boolean);
      // A summary groups text entries only; all raw boxes and vehicle totals remain intact.
      // Ignore malformed/partial groups instead of dropping their source observations.
      if (members.length !== group.track_ids.length) continue;
      for (const member of members) displayed.add(String(member.id));
      readings.push({...representative, text:group.normalized_text || group.text || representative.text,
        ocr_confidence:group.ocr_confidence ?? representative.ocr_confidence,
        first_seen:group.first_seen ?? representative.first_seen, last_seen:group.last_seen ?? representative.last_seen,
        observations:group.observations ?? representative.observations,
        reading_group:group, grouped_track_count:members.length});
    }
    readings.push(...tracks.filter(track => !displayed.has(String(track.id))));
    return readings.sort((a,b) => a.first_seen - b.first_seen);
  }
  function controls() {
    $('detect-button').disabled = ['uploading', 'queued', 'processing'].includes(phase);
    $('detect-button').textContent = phase === 'paused' ? 'Resume video processing' : 'Process video again';
    $('stop-button').disabled = !['processing', 'queued'].includes(phase);
    $('stop-button').textContent = 'Pause';
    $('video-profile').disabled = ['uploading', 'queued', 'processing', 'paused'].includes(phase);
    window.PlateReID?.setBusy($('video-profile').disabled);
    $('read-text').disabled = $('video-profile').disabled;
    $('angle-correction').disabled = $('video-profile').disabled || $('mode').value === 'browser';
    $('ocr-model').disabled = $('video-profile').disabled || !wantsText() || $('mode').value === 'browser';
    $('green-filter').disabled = $('ocr-model').disabled;
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
    $('read-text').disabled = false; $('angle-correction').disabled = $('mode').value === 'browser';
    $('stage-timing').textContent = '';
    $('ocr-model').disabled = !wantsText() || $('mode').value === 'browser'; $('video-profile').disabled = false; $('green-filter').disabled = $('ocr-model').disabled; $('latency-title').textContent = 'Frame latency'; $('plate-count-note').textContent = 'Current analysed frame'; $('ocr-note').textContent = wantsText() ? 'Scores are model estimates, not verified accuracy' : 'Off · locating plates without reading letters';
  }
  async function json(response) {
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Recording request failed.');
    return result;
  }
  function plateEvidence(track) {
    const unavailable = () => {
      const placeholder = document.createElement('div'); placeholder.className = 'crop-unavailable';
      placeholder.textContent = 'Crop unavailable'; return placeholder;
    };
    if (typeof track.crop_url !== 'string' || !track.crop_url.startsWith('/api/')) return unavailable();
    const image = document.createElement('img'); image.src = base + track.crop_url;
    image.alt = `${track.localization_requires_review ? 'Vehicle-crop plate candidate' : track.experimental_review_only ? 'Experimental candidate' : 'Plate track'} ${track.id}`;
    image.addEventListener('error', () => image.replaceWith(unavailable()), {once:true});
    return image;
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
      $('latency-note').textContent = 'Decode + plate/vehicle detection + tracking · final export pending';
    }
    renderVehicleSummary(result.vehicle_summary, result.vehicle_tracks || []);
    renderAnalysisSummary(result.recognized_plate_summary, {plateCount:Array.isArray(result.tracks) ? result.tracks.length : null,
      vehicleSummary:result.vehicle_summary, scope:'recording', readText:result.read_text ?? wantsText(), pending:result.status !== 'complete'});
    window.PlateReID?.renderResult(result.reidentification);
    if (result.status !== 'complete') return;
    const player = $('annotated-video');
    player.src = base + result.video_url; player.hidden = false; player.load();
    $('preview').hidden = true; $('empty').hidden = true; $('video').hidden = true;
    $('video-export-actions').hidden = false;
    $('download-video').href = base + result.video_url + '?download=true';
    $('download-report').href = base + result.report_url;
    $('latency').textContent = `${result.processing_ms_per_frame.toFixed(1)} ms`;
    const readText = result.read_text ?? wantsText();
    $('latency-note').textContent = `Overall average/frame, including ${readText ? 'OCR, ' : ''}angle correction and MP4 export · plate detector p95 ${result.detection_ms?.p95 ?? '—'} ms`;
    $('stage-timing').textContent = `Plate detection p95: ${result.plate_detection_ms?.p95 ?? result.detection_ms?.p95 ?? '—'} ms · Vehicle detection p95: ${result.vehicle_detection_ms?.p95 ?? '—'} ms · Angle correction total: ${result.rectification_total_ms ?? 0} ms. Detection includes preprocessing and postprocessing. Parallel stages overlap; overall processing includes export.${vehiclePlateSearchTiming(result.vehicle_plate_search)}`;
    $('ocr-latency').textContent = readText ? `${Math.round(result.ocr_total_ms || 0)} ms` : 'Off';
    $('ocr-note').textContent = readText ? `Total OCR · ${result.ocr_engine || 'Plate reader'}` : 'Plate regions retained without reading text';
    $('plate-count').textContent = result.tracks.length;
    const experimental = result.tracks.filter(t => t.experimental_review_only).length;
    $('coverage').textContent = `${result.frames} frames searched · ${result.tracks.length} region tracks${experimental ? ` · ${experimental} experimental candidates` : ''}`;
    const vehicleMessage = result.vehicle_summary?.incomplete ? `Vehicle analysis incomplete. ${result.vehicle_summary.warning || 'Vehicle totals are unavailable.'}`
      : result.vehicle_summary?.available !== true ? 'Vehicle detection was unavailable.'
      : (result.vehicle_summary.total_vehicles ?? result.vehicle_tracks?.length ?? 0) > 0 ? 'Detected vehicle boxes are included.' : 'No vehicles were detected.';
    $('status').textContent = `Annotated video ready. ${result.detector_model || "Plate detector"} · Every frame was searched with ${result.engine}. ${result.processing_ms_per_frame < 40 ? 'Overall processing averaged under the 40 ms/frame target for this recording.' : 'The 40 ms/frame target was not achieved for this recording.'} Play or download below. ${readText ? 'Verify text against the original crops.' : 'Text reading was off.'} ${vehicleMessage}`;
    $('recording-log').hidden = false;
    const rows = $('recording-rows'); rows.replaceChildren();
    const cards = $('vehicle-cards'); cards.replaceChildren();
    const readings = recordingReadings(result, readText);
    const grouped = result.tracks.length - readings.length;
    if (grouped > 0) $('coverage').textContent += ` · ${grouped} repeat track${grouped === 1 ? '' : 's'} grouped by plate text`;
    for (const track of readings) {
      const row = rows.insertRow();
      row.insertCell().textContent = `${track.first_seen.toFixed(2)}–${track.last_seen.toFixed(2)} s`;
      const cropCell = row.insertCell(); cropCell.append(plateEvidence(track)); appendRectification(cropCell, track.rectification, track.rectified_crop_url, base);
      const readingCell = row.insertCell(); readingCell.textContent = readText ? track.text || 'Unreadable' : 'Plate region · text reading off';
      appendGreenEnhancement(readingCell, track.green_enhancement, base, false);
      appendOcrReview(readingCell, track.ocr_review || track.ocr_details);
      row.insertCell().textContent = track.grouped_track_count > 1 ? `${track.observations} frames · ${track.grouped_track_count} tracks` : track.observations;
      const detectionScore = track.detection_confidence ?? track.max_detection_confidence;
      const detectionLabel = Number.isFinite(detectionScore) ? `${Math.round(detectionScore * 100)}% detection` : 'Detection score unavailable';
      row.insertCell().textContent = track.localization_requires_review ? 'Vehicle-crop plate candidate · verify region' : !readText ? `${detectionLabel} · inspect crop` : track.experimental_review_only ? 'Candidate · experimental review' : track.reading_group?.status === 'ambiguous' ? 'Repeated text in simultaneous tracks · review each crop' : track.format_status === 'valid' && track.ocr_confidence >= .8 && !(track.ocr_review || track.ocr_details)?.requires_review ? `${Math.round(track.ocr_confidence*100)}% · verify crop` : 'Uncertain · review crop';
      const card = document.createElement('article'); card.className = 'vehicle-card';
      const copy = document.createElement('div'), title = document.createElement('span'), text = document.createElement('strong'), detail = document.createElement('small');
      title.className = 'vehicle-title';
      title.textContent = track.localization_requires_review ? `Plate candidate ${track.id}` : track.experimental_review_only ? `Candidate ${track.id} · experimental review` : track.grouped_track_count > 1 ? `Registration · ${track.grouped_track_count} track sightings` : `Plate track ${track.id}`;
      text.textContent = readText ? track.text || 'Unreadable' : 'Plate region';
      detail.textContent = `${track.observations} frames · ${detectionLabel}${readText ? ` · ${Math.round(track.ocr_confidence*100)}% OCR` : ' · text reading off'}${track.reading_group?.status === 'ambiguous' ? ' · repeated simultaneous text; review' : ''}`;
      if (track.localization_requires_review) detail.textContent = `Vehicle-crop plate candidate · verify region · ${detail.textContent}`;
      copy.append(title,text,detail); appendGreenEnhancement(copy, track.green_enhancement, base, false);
      appendOcrReview(copy, track.ocr_review || track.ocr_details);
      appendRectification(copy, track.rectification, track.rectified_crop_url, base);
      card.append(plateEvidence(track),copy); cards.append(card);
    }
    if (!result.tracks.length) {
      const cell = rows.insertRow().insertCell(); cell.colSpan = 5; cell.textContent = 'No plate regions detected in this recording.';
      cards.textContent = 'No plate regions found. Try the detailed profile for small or distant plates.';
    }
    const currentRows = $('result-rows'); currentRows.replaceChildren();
    const cell = currentRows.insertRow().insertCell(); cell.colSpan = 6; cell.className = 'empty-row';
    cell.textContent = grouped > 0 ? 'See recording detections below. Repeated trusted registrations share one entry; all plate tracks remain in the report and video.' : 'See the recording detections below for all plate tracks.';
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
    $('status').textContent = 'Uploading recording for plate regions, vehicle analytics and annotated MP4 export.';
    const form = new FormData(); form.append('video', selectedFile); form.append('profile', $('video-profile').value); form.append('ocr_model', $('ocr-model').value);
    form.append('read_text', String(wantsText()));
    form.append('angle_correction', String($('angle-correction').value === 'true'));
    form.append('green_filter', String(wantsText() && $('green-filter').value === 'emboss'));
    try {
      window.PlateReID?.appendForm(form);
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
  return {active, controls, reset, start, appendGreenEnhancement, appendRectification, appendOcrReview, renderVehicleSummary, renderAnalysisSummary, categoryLabel, vehiclePlateSearchTiming, pause:() => action('pause'),
    resume:() => phase === 'paused' ? action('resume') : start(file)};
})();
