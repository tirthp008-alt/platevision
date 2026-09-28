const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const {readFileSync}=require('node:fs');
class Element {
  constructor(){this.value='';this.textContent='';this.children=[];}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  insertRow(){const row=new Element();this.append(row);return row;}
  insertCell(){const cell=new Element();this.append(cell);return cell;}
}
function client(fetch=async()=>({ok:true,json:async()=>({})})){
  const elements=new Map(),element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  element('mode').value='api';element('reidentify').value='false';element('api-url').value='http://127.0.0.1:8003';
  const context={window:{},document:{getElementById:element,createElement:()=>new Element()},fetch,AbortSignal,Date};
  vm.runInNewContext(readFileSync(require.resolve('../reid-ui.js'),'utf8'),context);
  return {api:context.window.PlateReID,element};
}

test('off mode sends no ReID fields or invented capture metadata',()=>{
  const {api,element}=client(),form=new FormData();
  api.appendForm(form);assert.equal([...form.entries()].length,0);
  assert.equal(element('camera-id').value,'');assert.equal(element('observed-at').value,'');
});

test('enabled matching requires real metadata and sends explicitly entered UTC timestamp',()=>{
  const {api,element}=client();element('reidentify').value='true';
  assert.throws(()=>api.appendForm(new FormData()),/camera ID/);
  element('camera-id').value=' gate-A ';assert.throws(()=>api.appendForm(new FormData()),/actual photo time/);
  element('observed-at').value='2026-09-27T10:30:15';const form=new FormData();api.appendForm(form);
  assert.equal(form.get('reidentify'),'true');assert.equal(form.get('camera_id'),'gate-A');
  assert.equal(form.get('observed_at'),'2026-09-27T10:30:15.000Z');
});

test('saving a camera route preserves other routes and does not save placeholder values',async()=>{
  const calls=[];
  const {element}=client(async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>options?.method==='PUT'?JSON.parse(options.body):{routes:[{source_camera:'X',target_camera:'Y',min_seconds:10,max_seconds:100}]}};});
  element('route-save').onclick();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.length,0);assert.match(element('route-status').textContent,/Enter two different/);
  for(const [id,value]of Object.entries({'route-from':'A','route-to':'B','route-min':'60','route-max':'600'}))element(id).value=value;
  element('route-save').onclick();await new Promise(resolve=>setImmediate(resolve));
  const sent=JSON.parse(calls[1].options.body);assert.equal(sent.routes.length,2);
  assert.equal(sent.routes[0].source_camera,'X');assert.equal(sent.routes[1].min_seconds,60);
  assert.equal(calls[1].options.method,'PUT');
});

test('candidates remain unverified, use decimal scores, and do not alter vehicle counts',()=>{
  const {api,element}=client();element('vehicle-count').textContent='7';
  api.renderResult({model:'Prototype',processing_time_ms:42,observations:[{camera_id:'B',track_id:2,status:'candidate',selected_candidate:'one',
    matches:[{observation_id:'one',camera_id:'A',track_id:1,appearance_similarity:.82,review_score:.79,travel_seconds:120,plate_evidence:'not_available'}]}]});
  const cells=element('reid-rows').children[0].children;
  assert.equal(cells[2].textContent,'Preferred candidate · unverified');
  assert.equal(cells[3].textContent,'0.820');assert.equal(cells[4].textContent,'0.790');
  assert.equal(element('vehicle-count').textContent,'7');assert.equal(element('reid-latency').textContent,'42.0 ms');
});

test('ReID error is visible without replacing original analysis counts',()=>{
  const {api,element}=client();element('plate-count').textContent='10';
  api.renderResult({status:'error',warning:'Appearance model unavailable',observations:[]});
  assert.match(element('reid-result-note').textContent,/Appearance model unavailable/);
  assert.match(element('reid-rows').children[0].children[0].textContent,/Plate and vehicle results remain available/);
  assert.equal(element('plate-count').textContent,'10');
});

test('analysis locks camera metadata and optional matching controls',()=>{
  const {api,element}=client();element('reidentify').value='true';api.setBusy(true);
  assert.equal(element('camera-id').disabled,true);assert.equal(element('observed-at').disabled,true);
  assert.equal(element('reidentify').disabled,true);api.setBusy(false);assert.equal(element('camera-id').disabled,false);
});

test('actual A-to-B API candidates show the source, preferred selection, UTC time and travel gate',()=>{
  const {api,element}=client();
  const sourceTime=Date.parse('2026-09-27T10:30:00Z')/1000;
  const candidate={source_observation_id:'source-one',source_camera:'A',source_track_id:'image:1',
    source_observed_at:sourceTime,source_last_seen_at:sourceTime,appearance_similarity:.91,
    review_score:.95,travel_seconds:180,min_seconds:60,max_seconds:600,
    plate_evidence:{kind:'agreement',known_matching_characters:5,boost:.04}};
  api.renderResult({status:'complete',note:'Review candidates only.',observations:[{
    camera_id:'B',track_id:'image:2',observed_at:sourceTime+180,status:'candidate',
    selected_candidate:candidate,candidates:[candidate]}]});
  const cells=element('reid-rows').children[0].children;
  assert.equal(cells[0].textContent,'B · track image:2');
  assert.equal(cells[0].children[0].textContent,'2026-09-27 10:33:00 UTC');
  assert.equal(cells[1].textContent,'A · track image:1');
  assert.equal(cells[1].children[0].textContent,'2026-09-27 10:30:00 UTC');
  assert.equal(cells[2].textContent,'Preferred candidate · unverified');
  assert.equal(cells[5].textContent,'180.0 s');
  assert.equal(cells[5].children[0].textContent,'Allowed 60–600 s');
  assert.equal(cells[6].textContent,'5 characters agree · +0.040 score');
  assert.match(element('reid-result-note').textContent,/Review candidates only/);
});

test('no readable plate leaves the appearance candidate usable and explains missing route gates',()=>{
  const {api,element}=client();
  api.renderResult({observations:[{camera_id:'B',track_id:'2',status:'candidate',matches:[{
    source_camera:'A',source_track_id:'1',appearance_similarity:.9,plate_evidence:{kind:'none'}}]},
    {camera_id:'C',track_id:'3',status:'new',rejected_candidates:{no_route:2,chronological_or_travel:1}}]});
  assert.equal(element('reid-rows').children[0].children[6].textContent,'No plate-text evidence');
  assert.match(element('reid-rows').children[1].children[2].textContent,/no camera route; capture order or travel time/);
});

test('backend failed status remains visible while preserving the main analysis',()=>{
  const {api,element}=client();element('vehicle-count').textContent='2';
  api.renderResult({status:'failed',warning:'Appearance matching failed.',observations:[]});
  assert.match(element('reid-rows').children[0].children[0].textContent,/Plate and vehicle results remain available/);
  assert.equal(element('vehicle-count').textContent,'2');
});

test('saving new travel gates removes candidate rows computed with old routes',async()=>{
  const {api,element}=client(async(url,options)=>({ok:true,json:async()=>options?.method==='PUT'?JSON.parse(options.body):{routes:[]}}));
  api.renderResult({observations:[{camera_id:'B',track_id:2,matches:[{source_camera:'A',source_track_id:1}]}]});
  for(const [id,value]of Object.entries({'route-from':'A','route-to':'B','route-min':'200','route-max':'600'}))element(id).value=value;
  element('route-save').onclick();await new Promise(resolve=>setImmediate(resolve));
  assert.match(element('reid-result-note').textContent,/Camera routes changed/);
  assert.match(element('reid-rows').children[0].children[0].textContent,/No cross-camera candidates/);
});
