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
  setAttribute() {} after() {} pause() {} load() {} removeAttribute() {}
  async play() { this.paused=false; }
  toBlob(callback) { callback(new Blob(['frame'],{type:'image/jpeg'})); }
  getContext() {
    if (!this.context) {
      this.drawCalls=[];this.rectangles=[];this.rectangleStyles=[];this.labels=[];
      this.context={drawImage:(...args)=>this.drawCalls.push(args),clearRect() {},strokeRect:(...args)=>{this.rectangles.push(args);this.rectangleStyles.push(this.context.strokeStyle);},
        fillRect() {},fillText:(label)=>this.labels.push(label),measureText:()=>({width:80})};
    }
    return this.context;
  }
}
function client(result, loadApp = false, search = '?engine=api&detector=regions') {
  const elements = new Map(), requests = [];
  const document = {createElement:tag => new Element(tag), getElementById:id => {
    if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);
  }};
  const element = id => document.getElementById(id);
  element('mode').value = 'api'; element('api-url').value = 'http://127.0.0.1:8003';
  element('read-text').value = 'false'; element('angle-correction').value = 'true'; element('green-filter').value = 'emboss';
  const context = {window:{addEventListener() {}}, document, fetch:async (url, options) => {
    requests.push({url, form:options?.body}); return {ok:true, json:async () => result};
  }, FormData, AbortSignal, AbortController, clearTimeout, setTimeout, performance,
    location:{search}, URLSearchParams, setInterval:() => 0,
    URL:class extends URL {static createObjectURL() {return 'blob:fixture';} static revokeObjectURL() {}},
    Image:class {constructor() {this.naturalWidth=100;this.naturalHeight=50;} set src(_value) {this.onload();}},
    PlateVideo:require('../video-analysis.js')};
  vm.runInNewContext(readFileSync(require.resolve('../video-export.js'), 'utf8'), context);
  context.PlateExport = context.window.PlateExport;
  if (loadApp) { context.setTimeout = () => 0; vm.runInNewContext(readFileSync(require.resolve('../app.js'), 'utf8'), context); }
  return {api:context.PlateExport, element, requests, context};
}
const frameSummary = {available:true, scope:'frame', total_vehicles:3,
  counts_by_category:{car:1,two_wheeler:2,bus:0,truck:0,bicycle:0,other:0}};

test('launch queries select the local API and reject credentials, paths, and remote endpoints', () => {
  for (const [search, expected] of [
    ['', 'http://127.0.0.1:8003'],
    ['?detector=green', 'http://127.0.0.1:8003'],
    ['?detector=regions', 'http://127.0.0.1:8003'],
    ['?detector=nano', 'http://127.0.0.1:8004'],
    ['?detector=yolo11n', 'http://127.0.0.1:8004'],
    ['?api=http%3A%2F%2F127.0.0.1%3A8765', 'http://127.0.0.1:8765'],
    ['?api=https%3A%2F%2Flocalhost%3A8765', 'https://localhost:8765'],
    ['?api=not-a-url', 'http://127.0.0.1:8003'],
    ['?api=https%3A%2F%2Fexample.com', 'http://127.0.0.1:8003'],
    ['?api=http%3A%2F%2Fuser%3Asecret%40localhost%3A8765', 'http://127.0.0.1:8003'],
    ['?api=http%3A%2F%2Flocalhost%3A8765%2Fprivate', 'http://127.0.0.1:8003'],
    ['?api=http%3A%2F%2Flocalhost%3A8765%3Fx%3D1', 'http://127.0.0.1:8003'],
    ['?api=http%3A%2F%2Flocalhost%3A8765%23fragment', 'http://127.0.0.1:8003'],
  ]) {
    assert.equal(client({}, true, search).element('api-url').value, expected, search);
  }
});

test('photo localization keeps unreadable regions, sends no OCR and reports real vehicle counts', async () => {
  const result = {read_text:false, processing_time_ms:38, ocr_time_ms:0, detections:[
    {bounding_box:{x:10,y:10,width:80,height:20}, normalized_text:'', ocr_confidence:null, detection_confidence:.83,
      rectification:{applied:true}, rectified_crop_url:'/api/results/example/plate/rectified'}],
    vehicle_summary:frameSummary, vehicles:[{id:'vehicle-1',category:'car',label:'Car',confidence:.91,bounding_box:{x:1,y:1,width:90,height:45},plate_ids:['plate-1']}]};
  const {element, requests} = client(result, true);
  element('read-text').value = 'false';
  const image = new Blob(['fixture'], {type:'image/jpeg'}); image.name = 'fixture.jpg';
  await element('media-input').handlers.change({target:{files:[image]}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests[0].url, 'http://127.0.0.1:8003/api/detect/image');
  assert.equal(requests[0].form.get('read_text'), 'false');
  assert.equal(requests[0].form.get('angle_correction'), 'true');
  assert.equal(requests[0].form.get('green_filter'), 'false');
  assert.equal(element('plate-count').textContent, 1);
  assert.equal(element('vehicle-count').textContent, 3);
  assert.equal(element('ocr-latency').textContent, 'Off');
  assert.equal(element('green-filter').disabled, true);
  assert.equal(element('ocr-model').disabled, true);
  const cells = element('result-rows').children[0].children;
  assert.equal(cells[2].children[0].textContent, 'Plate region');
  assert.equal(cells[3].textContent, '83%');
  assert.equal(cells[4].textContent, 'Not requested');
  assert.equal(cells[5].textContent, 'Detected region · inspect crop');
  assert.equal(cells[1].children.length, 2, 'original plus separate corrected view');
  assert.match(element('latency-note').textContent, /includes network \+ plate\/vehicle detection/);
});

test('image analysis enables OCR and angle correction by default while keeping the off option', async () => {
  const result = {read_text:true,processing_time_ms:50,ocr_time_ms:15,detections:[{
    bounding_box:{x:10,y:10,width:80,height:20},normalized_text:'GJ01AB1234',ocr_confidence:.94,
    format_status:'valid',detection_confidence:.91}],vehicles:[],vehicle_summary:frameSummary,
    recognized_plate_summary:{enabled:true,scope:'frame',unique_registrations:1,unresolved_track_count:0,
      ambiguous_track_count:0,suppressed_repeat_count:0}};
  const {element,requests} = client(result,true);
  assert.equal(element('read-text').value,'true');
  assert.equal(element('angle-correction').value,'true');
  const image=new Blob(['fixture'],{type:'image/jpeg'});image.name='fixture.jpg';
  await element('media-input').handlers.change({target:{files:[image]}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(requests[0].form.get('read_text'),'true');
  assert.equal(requests[0].form.get('angle_correction'),'true');
  assert.equal(element('result-rows').children[0].children[2].children[0].textContent,'GJ01AB1234');
  assert.equal(element('ocr-latency').textContent,'15 ms');
  assert.deepEqual(element('recognition-breakdown').children.map(card=>card.children[1].textContent),[1,1,0,0,3,0]);
  assert.equal(element('analysis-summary-scope').textContent,'CURRENT FRAME');
});

test('video localization keeps region tracks and separates recording counts from current visibility', async () => {
  const result = {id:'fixture',status:'complete',read_text:false,processing_ms_per_frame:40,detection_ms:{p95:20},ocr_total_ms:0,
    frames:20,video_url:'/api/videos/fixture/result',report_url:'/api/videos/fixture/report',
    vehicle_plate_search:{scope:'recording',searched_vehicles:20,added_regions:3,time_ms:12.3},
    vehicle_summary:{...frameSummary,scope:'recording',total_vehicles:5,visible_now:2,max_visible:3,confirmed_tracks:4,tentative_tracks:1},
    vehicle_tracks:[{id:1,category:'car',confidence:.9,observations:20,confirmed:true}],
    tracks:[{id:1,first_seen:0,last_seen:1,crop_url:'/api/videos/fixture/crops/1',text:'',ocr_confidence:null,observations:20,max_detection_confidence:.88,
      rectification:{applied:true},rectified_crop_url:'/api/videos/fixture/crops/1?variant=rectified'}]};
  const {api,element,requests} = client(result);
  await api.start(new Blob(['fixture'],{type:'video/mp4'}));
  assert.equal(requests[0].form.get('read_text'),'false');
  assert.equal(requests[0].form.get('angle_correction'),'true');
  assert.equal(requests[0].form.get('green_filter'),'false');
  assert.equal(element('vehicle-count').textContent,5);
  assert.match(element('vehicle-note').textContent,/tracks.*may fragment/);
  assert.match(element('vehicle-summary-note').textContent,/2 visible.*3 maximum simultaneously.*4 confirmed/);
  const cells=element('recording-rows').children[0].children;
  assert.equal(cells[2].textContent,'Plate region · text reading off');
  assert.equal(cells[4].textContent,'88% detection · inspect crop');
  assert.equal(cells[1].children.length,2);
  assert.equal(element('ocr-latency').textContent,'Off');
  assert.match(element('stage-timing').textContent,/Vehicle crop search: 20 per-frame checks · 3 additional plate observations · 12.3 ms/);
});

test('unavailable vehicle model is not falsely reported as zero or inferred from plate count', () => {
  const {api,element}=client({});
  api.renderVehicleSummary({available:false,scope:'frame',total_vehicles:null},[]);
  assert.equal(element('vehicle-count').textContent,'—');
  assert.equal(element('vehicle-breakdown').children[0].children[1].textContent,'—');
  assert.match(element('vehicle-summary-note').textContent,/unavailable/);
});

test('angle evidence rejects external preview URLs and unapplied corrections', () => {
  const {api}=client({}),parent=new Element();
  api.appendRectification(parent,{applied:true},'https://example.invalid/private','http://127.0.0.1:8003');
  api.appendRectification(parent,{applied:false},'/api/results/fixture','http://127.0.0.1:8003');
  assert.equal(parent.children.length,0);
});

test('vehicle categories follow model support and keep tempo traveller distinct from cargo vehicles', () => {
  const {api,element}=client({});
  api.renderVehicleSummary({...frameSummary,supported_categories:['car','three_wheeler','lcv','tempo','van'],
    unsupported_categories:[],counts_by_category:{car:1,three_wheeler:2,lcv:3,tempo:4,van:0}},[]);
  const cards=element('vehicle-breakdown').children;
  assert.equal(cards.length,5);
  assert.equal(cards[2].children[0].textContent,'Light commercial / cargo tempo');
  assert.equal(cards[2].children[1].textContent,3);
  assert.equal(cards[3].children[0].textContent,'Tempo Travellers');
  assert.equal(cards[3].children[1].textContent,4);
  api.renderVehicleSummary({...frameSummary,supported_categories:['car'],unsupported_categories:['tempo','auto_rickshaw']},[]);
  assert.equal(element('vehicle-breakdown').children.length,1);
  assert.match(element('vehicle-capability-note').textContent,/Not classified separately.*Tempo Traveller.*Auto-rickshaw/);
});

test('video missing crop and failed vehicle analysis show unavailable evidence and incomplete counts', async () => {
  const result={id:'fixture',status:'complete',read_text:false,processing_ms_per_frame:59,detection_ms:{p95:26},frames:20,
    video_url:'/api/videos/fixture/video',report_url:'/api/videos/fixture/report',
    vehicle_summary:{...frameSummary,scope:'recording',incomplete:true,available:false,
      warning:'Vehicle analysis stopped after an inference error. Plate capture continued; vehicle totals are unavailable.'},
    vehicle_tracks:[],tracks:[{id:1,first_seen:0,last_seen:1,crop_url:null,text:'',ocr_confidence:null,observations:20,detection_confidence:.8}]};
  const {api,element}=client(result);
  await api.start(new Blob(['fixture'],{type:'video/mp4'}));
  const evidence=element('recording-rows').children[0].children[1].children[0];
  assert.equal(evidence.tagName,'div');
  assert.equal(evidence.textContent,'Crop unavailable');
  assert.equal(evidence.src,undefined);
  assert.equal(element('vehicle-cards').children[0].children[0].textContent,'Crop unavailable');
  assert.equal(element('vehicle-count').textContent,'—');
  assert.match(element('vehicle-summary-note').textContent,/Incomplete.*inference error/);
  assert.equal(element('vehicle-scope').textContent,'INCOMPLETE');
  assert.match(element('status').textContent,/40 ms\/frame target was not achieved/);
  assert.match(element('status').textContent,/Vehicle analysis incomplete/);
  assert.doesNotMatch(element('status').textContent,/vehicle boxes are/i);
  assert.match(element('stage-timing').textContent,/Plate detection p95.*Vehicle detection p95/);
  assert.doesNotMatch(element('stage-timing').textContent,/inference p95/);
});

test('complete video with no vehicles does not claim vehicle boxes exist', async () => {
  const result={id:'fixture',status:'complete',read_text:false,processing_ms_per_frame:59,detection_ms:{p95:26},frames:20,
    video_url:'/api/videos/fixture/video',report_url:'/api/videos/fixture/report',tracks:[],vehicle_tracks:[],
    vehicle_summary:{...frameSummary,scope:'recording',total_vehicles:0}};
  const {api,element}=client(result);
  await api.start(new Blob(['fixture'],{type:'video/mp4'}));
  assert.match(element('status').textContent,/No vehicles were detected/);
  assert.doesNotMatch(element('status').textContent,/vehicle boxes are/i);
});

test('live frame requests never trigger expensive optional cross-camera embedding', async () => {
  const {element,requests,context}=client({read_text:false,detections:[],vehicles:[],processing_time_ms:40},true);
  context.window.PlateReID={enabled:()=>true,setBusy(){},renderResult(){},appendForm(){throw new Error('ReID must not run on each live frame');}};
  context.navigator={mediaDevices:{getUserMedia:async()=>({getTracks:()=>[]})}};
  Object.assign(element('video'),{videoWidth:100,videoHeight:50,readyState:2,currentTime:1});
  await element('camera-button').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(requests.length,1);
  assert.match(requests[0].url,/\/api\/detect\/frame$/);
  assert.equal(requests[0].form.has('reidentify'),false);
  assert.equal(requests[0].form.has('camera_id'),false);
  assert.match(element('status').textContent,/Cross-camera matching is off for live-camera frames/);
});

test('recording groups repeated trusted text while retaining simultaneous and uncertain sightings', async () => {
  const track=(id,text,first,last,confidence=.95)=>({id,text,first_seen:first,last_seen:last,ocr_confidence:confidence,
    format_status:confidence>=.8?'valid':'uncertain',observations:10,detection_confidence:.9,crop_url:`/api/videos/fixture/crops/${id}`});
  const result={id:'fixture',status:'complete',read_text:true,processing_ms_per_frame:55,frames:60,
    video_url:'/api/videos/fixture/video',report_url:'/api/videos/fixture/report',
    vehicle_summary:{...frameSummary,scope:'recording',total_vehicles:5},vehicle_tracks:[],
    tracks:[track(1,'GJ01AB1234',0,1,.91),track(2,'GJ01AB1234',3,4,.98),track(3,'KA25HB6656',1,3),
      track(4,'KA25HB6656',2,4),track(5,'',4,5,0),track(6,'GJ01AB1234',5,6,.5)],
    recognized_plate_summary:{enabled:true,unique_registrations:1,unresolved_track_count:2,ambiguous_track_count:2,suppressed_repeat_count:1,groups:[
      {text:'GJ01AB1234',representative_track_id:2,track_ids:[1,2],first_seen:0,last_seen:4,observations:20,ocr_confidence:.98,status:'recognized'},
      {text:'KA25HB6656',representative_track_id:3,track_ids:[3],status:'ambiguous'},
      {text:'KA25HB6656',representative_track_id:4,track_ids:[4],status:'ambiguous'}],unresolved_track_ids:[5,6]}};
  const original=JSON.stringify(result);
  const {api,element}=client(result);element('read-text').value='true';
  await api.start(new Blob(['fixture'],{type:'video/mp4'}));
  const rows=element('recording-rows').children;
  assert.equal(rows.length,5);
  assert.equal(element('vehicle-cards').children.length,5);
  assert.equal(element('plate-count').textContent,6);
  assert.equal(element('vehicle-count').textContent,5);
  assert.equal(rows[0].children[2].textContent,'GJ01AB1234');
  assert.match(rows[0].children[1].children[0].src,/\/crops\/2$/);
  assert.equal(rows[0].children[3].textContent,'20 frames · 2 tracks');
  assert.match(rows[1].children[4].textContent,/simultaneous tracks/);
  assert.match(rows[2].children[4].textContent,/simultaneous tracks/);
  assert.equal(rows[3].children[2].textContent,'Unreadable');
  assert.equal(rows[4].children[2].textContent,'GJ01AB1234');
  assert.equal(rows[4].children[4].textContent,'Uncertain · review crop');
  assert.match(element('coverage').textContent,/1 repeat track grouped/);
  assert.deepEqual(element('recognition-breakdown').children.map(card=>card.children[1].textContent),[6,1,2,2,5,1]);
  assert.equal(element('analysis-summary-scope').textContent,'WHOLE RECORDING');
  assert.equal(JSON.stringify(result),original,'grouping must not mutate the downloadable report or source tracks');
});

test('same-frame registrations keep separate physical regions and expose conflicting OCR cleanup', async () => {
  const detection=(x,review)=>({bounding_box:{x,y:10,width:35,height:20},normalized_text:'GJ01AB1234',
    ocr_confidence:.92,format_status:review?'uncertain':'valid',detection_confidence:.95,ocr_review:review});
  const review={requires_review:true,conflicting_readings:true,selected_method:'contrast_cleanup',
    attempts:[{method:'original',text:'GJ01AB1234',confidence:.52},{method:'contrast_cleanup',text:'GJ01AB1284',confidence:.92}]};
  const result={read_text:true,processing_time_ms:75,ocr_time_ms:40,
    detections:[detection(0),detection(40),detection(80,review)],vehicles:[],vehicle_summary:frameSummary};
  const {element}=client(result,true);
  const image=new Blob(['fixture'],{type:'image/jpeg'});image.name='fixture.jpg';
  await element('media-input').handlers.change({target:{files:[image]}});
  await new Promise(resolve=>setImmediate(resolve));
  const rows=element('result-rows').children;
  assert.equal(rows.length,3);
  assert.equal(element('vehicle-cards').children.length,3);
  assert.equal(element('plate-count').textContent,3);
  assert.match(rows[0].children[2].children[1].textContent,/2 simultaneous plate regions/);
  assert.match(rows[1].children[2].children[1].textContent,/2 simultaneous plate regions/);
  const cleanup=rows[2].children[2].children[1];
  assert.match(cleanup.children[0].textContent,/2 attempts.*review required/);
  assert.match(cleanup.children[1].textContent,/Readings disagree/);
  assert.equal(rows[2].children[2].children.length,2,'uncertain readings must not enter the trusted repeat count');
  assert.equal(rows[2].children[5].textContent,'Uncertain · review required');
});

test('backend conflict arrays only signal review when they contain differing readings',()=>{
  const {api}=client({}),parent=new Element();
  const original={method:'tight_original',text:'GJ01AB1234',confidence:.96};
  api.appendOcrReview(parent,{requires_review:false,conflicting_readings:[],selected_method:'tight_original',attempts:[original]});
  assert.equal(parent.children.length,0,'confident original OCR needs no cleanup disclosure');
  const attempts=[original,{...original,method:'grayscale_clahe'}];
  api.appendOcrReview(parent,{requires_review:false,conflicting_readings:[],selected_method:'grayscale_clahe',attempts});
  assert.equal(parent.children[0].children[0].textContent,'OCR cleanup · 2 attempts');
  assert.doesNotMatch(parent.children[0].children[1].textContent,/disagree/);
  api.appendOcrReview(parent,{requires_review:true,conflicting_readings:['GJ01AB1234','GJ01AB1284'],selected_method:'tight_original',attempts});
  assert.match(parent.children[1].children[0].textContent,/review required/);
  assert.match(parent.children[1].children[1].textContent,/Readings disagree/);
});

test('analysis summary separates raw tracks, trusted registrations, unresolved readings and vehicles',()=>{
  const {api,element}=client({});
  api.renderAnalysisSummary({enabled:true,unique_registrations:2,unresolved_track_count:3,
    ambiguous_track_count:4,suppressed_repeat_count:3},{scope:'recording',plateCount:12,readText:true,
    vehicleSummary:{available:true,total_vehicles:8,classification_review_count:2}});
  const cards=element('recognition-breakdown').children;
  assert.deepEqual(cards.map(card=>card.children[1].textContent),[12,2,3,4,8,3]);
  assert.equal(cards[0].children[0].textContent,'Plate region tracks');
  assert.equal(cards[1].children[0].textContent,'Trusted distinct registrations');
  assert.equal(cards[4].children[0].textContent,'Vehicle tracks');
  assert.equal(element('analysis-summary-scope').textContent,'WHOLE RECORDING');
  assert.match(element('recognition-summary-note').textContent,/may include fragments/);
  assert.match(element('recognition-summary-note').textContent,/Accuracy requires labelled/);
  assert.match(element('recognition-summary-note').textContent,/2 vehicle categories need review/);

  api.renderAnalysisSummary({enabled:false,unique_registrations:0},{scope:'frame',plateCount:5,readText:false,
    vehicleSummary:{available:false,total_vehicles:null}});
  const off=element('recognition-breakdown').children;
  assert.deepEqual(off.map(card=>card.children[1].textContent),[5,'—','—','—','—','—']);
  assert.equal(off[4].children[0].textContent,'Vehicles detected');
  assert.equal(element('analysis-summary-scope').textContent,'CURRENT FRAME');
  assert.match(element('recognition-summary-note').textContent,/OCR is off/);
});

test('vehicle category uncertainty is visible without reducing detection counts',()=>{
  const {api,element}=client({});
  api.renderVehicleSummary({...frameSummary,scope:'recording',classification_review_count:1},[
    {id:1,category:'bus',confidence:.95,classification_confidence:.81,detection_confidence:.98,observations:30,confirmed:true,
      classification_review:{requires_review:true,candidates:[{category:'bus',observations:17},{category:'car',observations:13}]}}]);
  const category=element('vehicle-rows').children[0].children[1];
  assert.equal(category.textContent,'Bus');
  assert.match(category.children[0].textContent,/Category needs review: Bus \(17 observations\); Car \(13 observations\)/);
  assert.equal(element('vehicle-rows').children[0].children[2].textContent,'81%');
  assert.equal(element('vehicle-count').textContent,3);
});

test('unreadable plates use tight detector rectangles while keeping padded crops and actual vehicle boxes',async()=>{
  const padded={x:5,y:8,width:40,height:20},tight={x:10,y:12,width:30,height:10};
  const secondTight={x:60,y:15,width:25,height:8};
  const vehicles=[{id:1,category:'car',confidence:.9,bounding_box:{x:0,y:0,width:48,height:45}},
    {id:2,category:'car',confidence:.9,bounding_box:{x:50,y:0,width:45,height:45}},
    {id:3,category:'car',confidence:.8,bounding_box:{x:5,y:30,width:35,height:20}}];
  const result={read_text:true,processing_time_ms:60,ocr_time_ms:20,vehicles,vehicle_summary:frameSummary,
    vehicle_plate_search:{searched_vehicles:2,added_regions:1,time_ms:8.25},detections:[
      {bounding_box:padded,tight_plate_box:tight,normalized_text:'',ocr_confidence:0,format_status:'uncertain',detection_confidence:.85},
      {bounding_box:{x:55,y:10,width:35,height:18},tight_plate_box:secondTight,normalized_text:'',ocr_confidence:0,format_status:'uncertain',detection_confidence:.81}]};
  const {element}=client(result,true);
  const image=new Blob(['fixture'],{type:'image/jpeg'});image.name='fixture.jpg';
  await element('media-input').handlers.change({target:{files:[image]}});
  await new Promise(resolve=>setImmediate(resolve));
  const rectangles=element('preview').rectangles;
  assert.equal(rectangles.length,5,'only three actual vehicle boxes and two actual plate boxes are drawn');
  assert.deepEqual(rectangles.slice(-2),[[10,12,30,10],[60,15,25,8]]);
  assert.equal(rectangles.some(rect=>JSON.stringify(rect)===JSON.stringify([5,8,40,20])),false,'padded crop must not become the plate outline');
  const row=element('result-rows').children[0].children;
  assert.equal(row[2].children[0].textContent,'Unreadable');
  const originalCrop=row[1].children[0].drawCalls[0][0];
  assert.deepEqual(originalCrop.drawCalls[0].slice(1,5),[5,8,40,20],'original evidence keeps its padded context');
  assert.equal(element('plate-count').textContent,2);
  assert.equal(element('vehicle-count').textContent,3);
  assert.match(element('stage-timing').textContent,/Vehicle crop search: 2 checks · 1 additional plate region · 8.3 ms/);
});

test('vehicle-crop plate proposals stay amber review candidates with strong OCR or OCR switched off',async()=>{
  for (const readText of [true,false]) {
    const result={read_text:readText,processing_time_ms:40,ocr_time_ms:10,vehicles:[],vehicle_summary:frameSummary,
      detections:[{bounding_box:{x:5,y:5,width:40,height:20},tight_plate_box:{x:8,y:8,width:34,height:14},
        normalized_text:'GJ01AB1234',ocr_confidence:.99,format_status:'valid',detection_confidence:.95,
        localization_requires_review:true,localization_source:'vehicle_crop'}]};
    const {element}=client(result,true);element('read-text').value=String(readText);
    const image=new Blob(['fixture'],{type:'image/jpeg'});image.name='fixture.jpg';
    await element('media-input').handlers.change({target:{files:[image]}});
    await new Promise(resolve=>setImmediate(resolve));
    assert.deepEqual(element('preview').rectangleStyles,['#e9bc73']);
    assert.equal(element('preview').labels[0],readText?'GJ01AB1234':'Plate candidate 1');
    const cells=element('result-rows').children[0].children;
    assert.equal(cells[5].textContent,'Vehicle-crop plate candidate · verify region');
    assert.equal(cells[1].children.length,1,'the candidate crop stays visible');
    assert.match(element('vehicle-cards').children[0].children[1].children[2].textContent,/Vehicle-crop plate candidate · verify region/);
    assert.equal(element('plate-count').textContent,1);
  }
});

test('video localization review survives readable OCR and keeps candidate crops',async()=>{
  const result={id:'fixture',status:'complete',read_text:true,processing_ms_per_frame:50,frames:10,
    video_url:'/api/videos/fixture/video',report_url:'/api/videos/fixture/report',vehicle_summary:frameSummary,
    tracks:[{id:1,text:'GJ01AB1234',ocr_confidence:.99,format_status:'valid',first_seen:0,last_seen:1,observations:10,
      crop_url:'/api/videos/fixture/crops/1',detection_confidence:.95,localization_source:'vehicle_crop',localization_requires_review:true}]};
  const {api,element}=client(result);element('read-text').value='true';
  await api.start(new Blob(['fixture'],{type:'video/mp4'}));
  const cells=element('recording-rows').children[0].children;
  assert.equal(cells[2].textContent,'GJ01AB1234');
  assert.equal(cells[4].textContent,'Vehicle-crop plate candidate · verify region');
  assert.equal(cells[1].children[0].alt,'Vehicle-crop plate candidate 1');
  assert.match(element('vehicle-cards').children[0].children[1].children[2].textContent,/Vehicle-crop plate candidate · verify region/);
  assert.equal(element('plate-count').textContent,1);
});
