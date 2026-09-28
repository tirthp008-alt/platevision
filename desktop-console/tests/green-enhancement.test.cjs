const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.value = ''; this.textContent = ''; this.handlers = {}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  insertRow() { const row = new Element('tr'); this.append(row); return row; }
  insertCell() { const cell = new Element('td'); this.append(cell); return cell; }
  cloneNode() { return Object.assign(new Element(this.tagName), {src:this.src, textContent:this.textContent}); }
  addEventListener(name, handler) { this.handlers[name] = handler; }
  setAttribute() {} after() {}
  getContext() { return {drawImage() {}, clearRect() {}, strokeRect() {}, fillRect() {}, fillText() {}, measureText:() => ({width:80})}; }
  pause() {} load() {} removeAttribute() {}
}

function client(fetch = async () => ({ok:true, json:async () => ({status:'failed', error:'Fixture'})})) {
  const elements = new Map();
  const document = {createElement:tag => new Element(tag), getElementById:id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  }};
  document.getElementById('mode').value = 'api';
  document.getElementById('api-url').value = 'http://127.0.0.1:8000';
  document.getElementById('green-filter').value = 'off';
  document.getElementById('read-text').value = 'true';
  document.getElementById('angle-correction').value = 'true';
  const context = {window:{}, document, fetch, FormData, AbortSignal, clearTimeout, setTimeout};
  vm.runInNewContext(readFileSync(require.resolve('../video-export.js'), 'utf8'), context);
  return {api:context.window.PlateExport, element:id => document.getElementById(id), context};
}

test('experimental alternative is separate, literal text and always requires review', () => {
  const {api} = client(); const parent = new Element();
  const original = new Element('strong'); original.textContent = 'KA25HB6656'; parent.append(original);
  const metadata = {applied:true, candidate_text:'<img onerror=alert(1)>', candidate_ocr_confidence:.99,
    preprocessing_ms:1, ocr_ms:20, crop_url:'/api/results/example/plate-emboss/crop'};
  const snapshot = JSON.stringify(metadata);
  api.appendGreenEnhancement(parent, metadata, 'http://127.0.0.1:8000');
  assert.equal(parent.children[0], original); assert.equal(original.textContent, 'KA25HB6656');
  assert.equal(JSON.stringify(metadata), snapshot);
  const details = parent.children[1];
  assert.match(details.children[0].textContent, /review required/);
  assert.match(details.children[1].textContent, /<img onerror=alert\(1\)>/);
  assert.equal(details.children.filter(node => node.tagName === 'img').length, 1);
  assert.equal(details.children[2].src, 'http://127.0.0.1:8000/api/results/example/plate-emboss/crop');
  assert.match(details.children[3].textContent, /21.0 ms/);
});

test('non-applied results stay hidden and video alternatives require no preview URL', () => {
  const {api} = client(); const parent = new Element();
  api.appendGreenEnhancement(parent, {applied:false}, 'http://localhost:8000');
  assert.equal(parent.children.length, 0);
  api.appendGreenEnhancement(parent, {applied:true, candidate_text:'', candidate_ocr_confidence:0}, 'http://localhost:8000', false);
  assert.match(parent.children[0].children[1].textContent, /Unreadable/);
  assert.equal(parent.children[0].children.filter(node => node.tagName === 'img').length, 0);
});

test('comparison preview never loads an external URL from returned metadata', () => {
  const {api} = client(); const parent = new Element();
  api.appendGreenEnhancement(parent, {applied:true, crop_url:'https://example.invalid/private'}, 'http://localhost:8000');
  assert.equal(parent.children[0].children.filter(node => node.tagName === 'img').length, 0);
});

test('video uploads send explicit off/on values and lock enhancement during upload', async () => {
  for (const [selection, expected] of [['off','false'], ['emboss','true']]) {
    let release; let form;
    const {api, element} = client((_url, options) => {
      form = options.body;
      return new Promise(resolve => { release = resolve; });
    });
    element('green-filter').value = selection;
    const upload = api.start(new Blob(['fixture'], {type:'video/mp4'}));
    assert.equal(form.get('green_filter'), expected);
    assert.equal(element('green-filter').disabled, true);
    release({ok:true, json:async () => ({id:'fixture', status:'failed', error:'Fixture completion'})});
    await upload;
    assert.equal(element('green-filter').disabled, false);
    element('mode').value = 'browser'; api.controls();
    assert.equal(element('green-filter').disabled, true);
  }
});

test('experimental video proposals stay candidates even when text resembles a valid plate', async () => {
  const result = {id:'fixture', status:'complete', processing_ms_per_frame:90, detection_ms:{p95:40},
    ocr_total_ms:15, frames:2, tracks:[{id:1, first_seen:0, last_seen:1, crop_url:'/api/videos/fixture/crops/1',
      text:'KA25HB6656', observations:2, format_status:'valid', ocr_confidence:.95, experimental_review_only:true}]};
  const {api, element} = client(async () => ({ok:true, json:async () => result}));
  await api.start(new Blob(['fixture'], {type:'video/mp4'}));
  assert.equal(element('recording-rows').children[0].children[2].textContent, 'KA25HB6656');
  assert.equal(element('recording-rows').children[0].children[4].textContent, 'Candidate · experimental review');
  assert.match(element('vehicle-cards').children[0].children[1].children[0].textContent, /experimental review/);
  assert.equal(element('vehicle-count').textContent, '—');
  assert.match(element('coverage').textContent, /1 experimental candidates/);
});

test('image API experimental flag reaches candidate titles and status without changing original OCR', async () => {
  const result = {processing_time_ms:80, detections:[{bounding_box:{x:10,y:10,width:80,height:20},
    normalized_text:'KA25HB6656', ocr_confidence:.95, format_status:'uncertain', detection_confidence:.4,
    experimental_review_only:true}], ocr_time_ms:20};
  const {context, element} = client(async () => ({ok:true, json:async () => result}));
  Object.assign(context, {PlateExport:context.window.PlateExport, PlateVideo:require('../video-analysis.js'),
    location:{search:'?engine=api&detector=nano'}, URLSearchParams, AbortController, performance,
    URL:{createObjectURL:() => 'blob:fixture', revokeObjectURL() {}}, setInterval:() => 0, setTimeout:() => 0,
    Image:class {constructor() {this.naturalWidth=100;this.naturalHeight=50;} set src(_value) {this.onload();}}});
  context.window.addEventListener = () => {};
  vm.runInNewContext(readFileSync(require.resolve('../app.js'), 'utf8'), context);
  element('read-text').value = 'true';
  const file = new Blob(['fixture'], {type:'image/jpeg'}); file.name='fixture.jpg';
  await element('media-input').handlers.change({target:{files:[file]}});
  await new Promise(resolve => setImmediate(resolve));
  const cells = element('result-rows').children[0].children;
  assert.match(cells[0].textContent, /Candidate #1 · experimental review/);
  assert.equal(cells[2].children[0].textContent, 'KA25HB6656');
  assert.equal(cells[4].textContent, '95%');
  assert.equal(cells[5].textContent, 'Candidate · experimental review');
  assert.match(element('vehicle-cards').children[0].children[1].children[0].textContent, /experimental review/);
  assert.equal(element('vehicle-count').textContent, '—');
  assert.equal(element('api-url').value, 'http://127.0.0.1:8004');
});
