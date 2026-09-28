/* Canonical desktop console client. */
(() => {
'use strict';
const $=id=>document.getElementById(id), preview=$('preview'), ctx=preview.getContext('2d'), video=$('video');
const frame=document.createElement('canvas'), fc=frame.getContext('2d',{willReadFrequently:true});
let imageFile=null;
let source=null,stream=null,url=null,generation=0,running=false,ready=false,busy=false,ocrBusy=false,workerPromise=null;
let sceneVehicles=[],vehicleSummary=null,plateSummary=null,tracks=[],nextId=1,lastFrame=-1,timer=null;const samples=[];
let recording=false,recordingNext=0,recordingDone=false,scanAbort=null,requestAbort=null;
const recordingReadings=new Map();
const wantsText=()=>$('read-text').value==='true';
$('read-text').value='true';$('angle-correction').value='true';
const parameters=new URLSearchParams(location.search);
$('mode').value=parameters.get('engine')==='browser'?'browser':'api';
$('ocr-model').value=parameters.get('ocr')==='resnet34'?'resnet34':'ppocr';
$('api-url').value='http://127.0.0.1:8003';
if(['regions','green'].includes(parameters.get('detector'))){
  $('api-url').value='http://127.0.0.1:8003';
  const note=document.createElement('p');note.className='hint';note.textContent='Plate regions + vehicle analytics. Detection confidence is a model score; accuracy is measured separately.';$('api-url').after(note);
}else if(['nano','yolo11n'].includes(parameters.get('detector'))){
  $('api-url').value='http://127.0.0.1:8004';
  const note=document.createElement('p');note.className='hint';
  note.textContent='YOLO11n comparison candidate: 20 training epochs. Validation improved, while reserved-test results regressed. Review detection and OCR separately.';
  $('api-url').after(note);
}
$('resolution').value=$('mode').value==='api'?'1920':'640';
$('ocr-model').disabled=!wantsText()||$('mode').value==='browser';$('green-filter').disabled=$('ocr-model').disabled;
const status=text=>{$('status').textContent=text;};
const valid=text=>/^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$/.test(text);
function overlap(a,b){const area=Math.max(0,Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y));return area/Math.max(1,a.width*a.height+b.width*b.height-area);}
function controls(){
 const active=running||busy;window.PlateReID?.setBusy(active);$('read-text').disabled=active;$('angle-correction').disabled=active||$('mode').value==='browser';
 $('ocr-model').disabled=active||!wantsText()||$('mode').value==='browser';$('green-filter').disabled=$('ocr-model').disabled;
 if(window.PlateExport?.active()){PlateExport.controls();return;}
 $('detect-button').disabled=!source||active||($('mode').value==='browser'&&!ready);$('detect-button').textContent=recording?(recordingDone?'Analyse recording again':'Resume recording analysis'):'Analyse image';$('stop-button').textContent=recording?'Pause':'Stop';$('stop-button').disabled=!running;$('video-interval').disabled=running;
}
function stop(clear=false){window.PlateExport?.reset();generation++;running=false;scanAbort?.abort();requestAbort?.abort();recording=false;recordingNext=0;recordingDone=false;recordingReadings.clear();$('recording-progress').hidden=true;$('recording-log').hidden=true;clearTimeout(timer);video.pause();stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.removeAttribute('src');video.load();video.hidden=true;if(url)URL.revokeObjectURL(url);url=null;source=null;imageFile=null;tracks=[];samples.length=0;lastFrame=-1;if(clear){sceneVehicles=[];vehicleSummary=null;plateSummary=null;$('media-input').value='';ctx.clearRect(0,0,preview.width,preview.height);$('empty').hidden=false;$('source-name').textContent='NO SOURCE';render();}controls();}

function videoEvent(event,signal){return new Promise((resolve,reject)=>{const cleanup=()=>{clearTimeout(timeout);video.removeEventListener(event,done);video.removeEventListener('error',failed);signal?.removeEventListener('abort',aborted);};const done=()=>{cleanup();resolve();};const failed=()=>{cleanup();reject(new Error('Cannot decode this recording. Try an MP4 or WebM.'));};const aborted=()=>{cleanup();reject(new DOMException('Analysis paused','AbortError'));};const timeout=setTimeout(()=>{cleanup();reject(new Error('Video frame took too long to load.'));},15000);video.addEventListener(event,done,{once:true});video.addEventListener('error',failed,{once:true});signal?.addEventListener('abort',aborted,{once:true});if(signal?.aborted)aborted();});}
function abortableDelay(ms,signal){return new Promise((resolve,reject)=>{const aborted=()=>{clearTimeout(timer);reject(new DOMException('Analysis paused','AbortError'));};const timer=setTimeout(()=>{signal.removeEventListener('abort',aborted);resolve();},ms);signal.addEventListener('abort',aborted,{once:true});if(signal.aborted)aborted();});}
async function requestModel(endpoint,form,signal){
  for(let attempt=0;attempt<=5;attempt++){
    const response=await fetch(endpoint,{method:'POST',body:form,signal:AbortSignal.any([signal,AbortSignal.timeout(30000)])});
    if(response.status!==429||!recording||attempt===5)return response;
    await abortableDelay(200*(attempt+1),signal);
  }
}
async function seekRecording(time,signal){video.pause();const target=time===0?Math.min(.001,video.duration/2):time;if(Math.abs(video.currentTime-target)<.0005&&video.readyState>=2)return;const waiting=videoEvent('seeked',signal);video.currentTime=target;await waiting;}
function timestamp(time){const minutes=Math.floor(time/60),seconds=(time%60).toFixed(1).padStart(4,'0');return `${String(minutes).padStart(2,'0')}:${seconds}`;}
function recordReadings(time){
  for(const track of tracks){
    if(wantsText()&&(!track.text||track.text.length<6))continue;
    const item=recordingReadings.get(track.id)||{text:track.text,first:time,last:time,count:0,confidence:0,crop:track.crop,formatStatus:track.formatStatus,detectionConfidence:track.detectionConfidence};
    item.last=time;item.count++;item.detectionConfidence=track.detectionConfidence;if(track.text!==item.text||track.confidence>=item.confidence){item.text=track.text;item.confidence=track.confidence;item.crop=track.crop;item.formatStatus=track.formatStatus;}
    recordingReadings.set(track.id,item);
  }
  $('recording-log').hidden=false;
  const rows=$('recording-rows');rows.replaceChildren();
  for(const item of recordingReadings.values()){
    const row=rows.insertRow();row.insertCell().textContent=`${timestamp(item.first)} – ${timestamp(item.last)}`;
    const img=document.createElement('img');img.src=item.crop.toDataURL('image/jpeg',.9);img.alt=`Recorded plate ${item.text}`;row.insertCell().append(img);
    row.insertCell().textContent=wantsText()?item.text:'Plate region · text reading off';row.insertCell().textContent=item.count;
    row.insertCell().textContent=!wantsText()?'Detected region · inspect crop':item.confidence>=70&&item.formatStatus==='valid'?'Format matched · verify crop':'Uncertain · review crop';
  }
  if(!recordingReadings.size){const cell=rows.insertRow().insertCell();cell.colSpan=6;cell.className='empty-row';cell.textContent=wantsText()?'No plate text has been read in the analysed frames.':'No plate regions found in the analysed frames.';}
}
async function analyseRecording(){
  if(busy||running||!recording)return;
  if(recordingDone){recordingNext=0;recordingDone=false;recordingReadings.clear();tracks=[];}
  const token=++generation,controller=new AbortController();scanAbort=controller;running=true;video.controls=false;controls();$('recording-progress').hidden=false;
  status('Analysing recording. Each selected frame waits for the model before advancing.');
  try{
    const completed=await PlateVideo.scan({duration:video.duration,interval:Number($('video-interval').value),from:recordingNext,
      active:()=>token===generation&&running,
      seek:time=>seekRecording(time,controller.signal),
      analyze:async time=>{if(!await detect())throw new Error('Frame analysis failed. Check the API, then resume the recording.');if(token===generation)recordReadings(time);},
      onProgress:progress=>{recordingNext=progress.nextTime;$('recording-meter').value=progress.index/progress.total*100;$('recording-position').textContent=`${timestamp(progress.time)} / ${timestamp(video.duration)} · ${progress.index} of ${progress.total} frames`;}});
    if(completed&&token===generation){recordingDone=true;status(`Recording complete. ${recordingReadings.size} plate region tracks${wantsText()?' with text':''}. Review the original crops.`);}
  }catch(error){if(token===generation&&error.name!=='AbortError')status(error.message);}
  finally{if(scanAbort===controller)scanAbort=null;if(token===generation){running=false;video.controls=true;controls();}}
}
function candidates(){const owned=[],mat=()=>{const m=new cv.Mat();owned.push(m);return m;};try{
 const src=cv.imread(frame);owned.push(src);const gray=mat(),edge=mat(),closed=mat(),hierarchy=mat(),contours=new cv.MatVector();owned.push(contours);
 cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);cv.Canny(gray,edge,70,170);const kernel=cv.getStructuringElement(cv.MORPH_RECT,new cv.Size(5,3));owned.push(kernel);cv.morphologyEx(edge,closed,cv.MORPH_CLOSE,kernel);cv.findContours(closed,contours,hierarchy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);
 const boxes=[],area=frame.width*frame.height;
 for(let i=0;i<contours.size();i++){const contour=contours.get(i);try{const b=cv.boundingRect(contour),ratio=b.width/b.height,size=b.width*b.height;if(b.width<38||b.height<12||ratio<1.4||ratio>6.8||size<area*.0005||size>area*.15)continue;const roi=edge.roi(b);let density;try{density=cv.countNonZero(roi)/size;}finally{roi.delete();}if(density<.08||density>.55)continue;boxes.push({...b,score:density*Math.sqrt(size)});}finally{contour.delete();}}
 boxes.sort((a,b)=>b.score-a.score);const kept=[];for(const b of boxes)if(!kept.some(k=>overlap(b,k)>.3)){kept.push(b);if(kept.length===20)break;}return kept;
}finally{owned.reverse().forEach(m=>m.delete());}}
function crop(box){const c=document.createElement('canvas'),scale=Math.min(3,360/box.width);c.width=Math.max(1,Math.round(box.width*scale));c.height=Math.max(1,Math.round(box.height*scale));c.getContext('2d').drawImage(frame,box.x,box.y,box.width,box.height,0,0,c.width,c.height);return c;}
function updateTracks(boxes){const used=new Set();tracks=boxes.map(box=>{const match=tracks.filter(t=>!used.has(t.id)&&overlap(t.box,box)>.4).sort((a,b)=>overlap(b.box,box)-overlap(a.box,box))[0];const t=match||{id:nextId++,text:'',confidence:0,attempts:0,lastOcr:0,votes:{}};used.add(t.id);t.box=box;t.crop=crop(box);return t;});}
function draw(vehicles=[]){
 preview.width=frame.width;preview.height=frame.height;ctx.drawImage(frame,0,0);
 const box=(b,label,color)=>{if(!b)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(b.x,b.y,b.width,b.height);const font=Math.max(14,Math.round(frame.width/85)),height=font+10;ctx.font=`600 ${font}px Arial`;const width=Math.min(frame.width,ctx.measureText(label).width+16),x=Math.max(0,Math.min(b.x,frame.width-width)),y=Math.max(height,b.y);ctx.fillStyle=color;ctx.fillRect(x,y-height,width,height);ctx.fillStyle='#102234';ctx.fillText(label,x+8,y-6);};
 vehicles.forEach((v,i)=>box(v.bounding_box,`${PlateExport.categoryLabel(v.category)} ${i+1} · ${Number.isFinite(v.confidence)?Math.round(v.confidence*100)+'%':'—'}`,'#7ec8ff'));
 tracks.forEach(t=>box(t.tightBox||t.box,t.localizationRequiresReview?(t.readText!==false&&t.text?t.text:`Plate candidate ${t.id}`):t.readText===false?`Plate ${t.id}${Number.isFinite(t.detectionConfidence)?' · '+Math.round(t.detectionConfidence)+'%':''}`:t.experimentalReviewOnly?`Candidate ${t.id} · review`:t.text||`Unverified ${t.id}`,t.localizationRequiresReview?'#e9bc73':t.readText===false?'#58dfa1':!t.experimentalReviewOnly&&t.text&&t.confidence>=80&&valid(t.text)&&!t.ocrReview?.requires_review?'#58dfa1':'#e9bc73'));
 $('empty').hidden=true;
}
function cropPreview(t){const c=document.createElement('canvas');c.width=t.crop.width;c.height=t.crop.height;c.getContext('2d').drawImage(t.crop,0,0);c.setAttribute('role','img');c.setAttribute('aria-label',`Plate region ${t.id}`);return c;}
function localizationReview(track){return track.localizationRequiresReview?(track.localizationSource==='vehicle_crop'?'Vehicle-crop plate candidate · verify region':'Plate candidate · verify region'):'';}
function repeatedReadings(){
 const counts=new Map();for(const track of tracks){if(track.readText===false||track.experimentalReviewOnly||track.localizationRequiresReview||track.confidence<80||track.formatStatus!=='valid'||track.ocrReview?.requires_review)continue;const key=(track.text||'').toUpperCase().replace(/[\s-]/g,'');if(valid(key))counts.set(key,(counts.get(key)||0)+1);}return counts;
}
// Launch links may choose another local port. External API addresses remain
// an explicit user choice in the address field, never a query-string redirect.
if(parameters.has('api')){
  try{
    const requestedApi=new URL(parameters.get('api'));
    if(['http:','https:'].includes(requestedApi.protocol)
      && ['127.0.0.1','localhost','[::1]'].includes(requestedApi.hostname)
      && !requestedApi.username && !requestedApi.password
      && requestedApi.pathname==='/' && !requestedApi.search && !requestedApi.hash){
      $('api-url').value=requestedApi.origin;
    }
  }catch{ /* Keep the selected model's default for malformed launch links. */ }
}
function appendRepeatedReading(parent,track,counts){
 if(track.readText===false||track.experimentalReviewOnly||track.localizationRequiresReview||track.confidence<80||track.formatStatus!=='valid'||track.ocrReview?.requires_review)return;
 const count=counts.get((track.text||'').toUpperCase().replace(/[\s-]/g,''))||0;if(count<2)return;
 const note=document.createElement('small');note.className='hint';note.textContent=`Same text in ${count} simultaneous plate regions · verify each crop`;parent.append(note);
}
function render(){
 renderVehicles();$('plate-count').textContent=tracks.length;const rows=$('result-rows');rows.replaceChildren();const repeated=repeatedReadings();
 PlateExport.renderAnalysisSummary(plateSummary,{plateCount:tracks.length,vehicleSummary,scope:'frame',readText:wantsText()});
 if(!tracks.length){const cell=rows.insertRow().insertCell();cell.colSpan=6;cell.className='empty-row';cell.textContent='No plate regions detected in the current frame.';return;}
 tracks.forEach(t=>{const row=rows.insertRow();row.insertCell().textContent=t.localizationRequiresReview?`Plate candidate #${t.id}`:t.experimentalReviewOnly?`Candidate #${t.id} · experimental review`:`Plate #${t.id}`;
 const cropCell=row.insertCell();cropCell.append(cropPreview(t));PlateExport.appendRectification(cropCell,t.rectification,t.rectifiedCropUrl,t.apiBase);
 const text=document.createElement('strong');text.textContent=t.readText===false?'Plate region':t.text||(t.attempts?'Unreadable':'Pending OCR');const readingCell=row.insertCell();readingCell.append(text);PlateExport.appendGreenEnhancement(readingCell,t.greenEnhancement,t.apiBase);PlateExport.appendOcrReview(readingCell,t.ocrReview);appendRepeatedReading(readingCell,t,repeated);
 row.insertCell().textContent=Number.isFinite(t.detectionConfidence)?`${Math.round(t.detectionConfidence)}%`:'—';row.insertCell().textContent=t.readText===false?'Not requested':t.confidence?`${Math.round(t.confidence)}%`:'—';
 row.insertCell().textContent=localizationReview(t)||(t.readText===false?'Detected region · inspect crop':t.experimentalReviewOnly?'Candidate · experimental review':t.text?(valid(t.text)&&t.formatStatus==='valid'&&t.confidence>=80&&!t.ocrReview?.requires_review?'Format matched · verify crop':'Uncertain · review required'):'Candidate · unverified');});
}
function renderVehicles(){
 PlateExport.renderVehicleSummary(vehicleSummary,sceneVehicles);
 const panel=$('vehicle-cards');panel.replaceChildren();const repeated=repeatedReadings();
 const readable=tracks.filter(t=>t.text&&!t.experimentalReviewOnly).length,experimental=tracks.filter(t=>t.experimentalReviewOnly).length;
 $('coverage').textContent=tracks.length?`${tracks.length} plate regions${wantsText()?` · ${readable} text readings`:''}${experimental?` · ${experimental} experimental candidates`:''}`:'Awaiting plate detections';
 if(!tracks.length){const empty=document.createElement('div');empty.className='vehicle-empty';empty.textContent='Plate crops appear here after analysing an image or video.';panel.append(empty);return;}
 tracks.forEach(t=>{const card=document.createElement('article');card.className='vehicle-card';const img=cropPreview(t),copy=document.createElement('div'),title=document.createElement('span');title.className='vehicle-title';title.textContent=t.localizationRequiresReview?`Plate candidate ${t.id}`:t.experimentalReviewOnly?`Candidate ${t.id} · experimental review`:`Plate region ${t.id}`;
 const reading=document.createElement('strong');reading.textContent=t.readText===false?'Plate region':t.text||(t.attempts?'Text unreadable':'Reading text…');const detail=document.createElement('small'),score=Number.isFinite(t.detectionConfidence)?Math.round(t.detectionConfidence)+'%':'—';
 detail.textContent=t.localizationRequiresReview?`${localizationReview(t)} · ${score} detection`:t.readText===false?`${score} detection confidence · text reading off`:t.experimentalReviewOnly?`Candidate · experimental review · ${score} detection · ${Math.round(t.confidence)}% OCR`:t.text?`${score} detection · ${Math.round(t.confidence)}% OCR · verify crop`:'Unverified region · review crop';
 copy.append(title,reading,detail);PlateExport.appendRectification(copy,t.rectification,t.rectifiedCropUrl,t.apiBase);PlateExport.appendGreenEnhancement(copy,t.greenEnhancement,t.apiBase,false);PlateExport.appendOcrReview(copy,t.ocrReview);appendRepeatedReading(copy,t,repeated);card.append(img,copy);panel.append(card);});
}
async function getWorker(){if(!workerPromise)workerPromise=(async()=>{if(!window.Tesseract)throw new Error('OCR library unavailable. Check your network and reload.');const worker=await Tesseract.createWorker('eng');await worker.setParameters({tessedit_char_whitelist:'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',tessedit_pageseg_mode:'6'});return worker;})().catch(e=>{workerPromise=null;throw e;});return workerPromise;}
async function recognize(token,all=false){if(ocrBusy)return;ocrBusy=true;try{const worker=await getWorker();if(token!==generation)return;const pending=tracks.filter(t=>!t.lastOcr||performance.now()-t.lastOcr>1200).sort((a,b)=>a.lastOcr-b.lastOcr).slice(0,all?20:2);for(const t of pending){if(token!==generation)break;const start=performance.now(),result=await worker.recognize(t.crop);if(token!==generation)break;t.lastOcr=performance.now();t.attempts++;const text=result.data.text.toUpperCase().replace(/[^A-Z0-9]/g,''),confidence=result.data.confidence;if(text.length>=6&&text.length<=12&&confidence>=45){t.votes[text]=(t.votes[text]||0)+confidence;const best=Object.keys(t.votes).sort((a,b)=>t.votes[b]-t.votes[a])[0];if(best===text){t.text=text;t.confidence=confidence;}}$('ocr-latency').textContent=`${Math.round(performance.now()-start)} ms`;render();}}catch(e){if(token===generation)status(e.message);}finally{ocrBusy=false;if(token!==generation&&source&&!running&&wantsText()&&$('mode').value==='browser')void recognize(generation,true);}}
async function detect(){if(busy||!source||($('mode').value==='browser'&&!ready))return false;const token=generation;busy=true;controls();const start=performance.now();try{const w=source.videoWidth||source.naturalWidth,h=source.videoHeight||source.naturalHeight;if(!w||!h)return;const scale=Math.min(1,Number($('resolution').value)/w);frame.width=Math.round(w*scale);frame.height=Math.round(h*scale);fc.drawImage(source,0,0,frame.width,frame.height);let vehicles=[],serverMs=null;
 if($('mode').value==='api'){const apiBase=$('api-url').value.replace(/\/$/,'');const blob=(!running&&imageFile&&scale===1)?imageFile:await new Promise(resolve=>frame.toBlob(resolve,'image/jpeg',.95));const form=new FormData();form.append('image',blob,imageFile?.name||'frame.jpg');form.append('profile',$('image-profile').value);form.append('ocr_model',$('ocr-model').value);if(!running&&!stream)window.PlateReID?.appendForm(form);form.append('read_text',String(wantsText()));form.append('angle_correction',String($('angle-correction').value==='true'));form.append('green_filter',String(wantsText()&&$('green-filter').value==='emboss'));requestAbort=new AbortController();const response=await requestModel(`${apiBase}/api/detect/${running?'frame':'image'}`,form,requestAbort.signal);const data=await response.json();serverMs=data.processing_time_ms;if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Model API rejected the frame.');if(token!==generation)return;updateTracks(data.detections.map(d=>d.bounding_box));data.detections.forEach((d,i)=>{tracks[i].readText=data.read_text??wantsText();if(tracks[i].readText)PlateVideo.reading(tracks[i],d.normalized_text,(d.ocr_confidence||0)*100,d.format_status,recording||!!stream);else{tracks[i].text='';tracks[i].confidence=0;}tracks[i].detectionConfidence=d.detection_confidence*100;tracks[i].attempts=1;tracks[i].vehicleId=d.vehicle_id;tracks[i].experimentalReviewOnly=!!d.experimental_review_only;tracks[i].localizationRequiresReview=!!d.localization_requires_review;tracks[i].localizationSource=d.localization_source||null;tracks[i].greenEnhancement=d.green_enhancement||null;tracks[i].ocrReview=d.ocr_review||d.ocr_details||null;tracks[i].apiBase=apiBase;tracks[i].tightBox=d.tight_plate_box;tracks[i].rectification=d.rectification;tracks[i].rectifiedCropUrl=d.rectified_crop_url;});window.PlateReID?.renderResult(data.reidentification);vehicles=data.vehicles||[];sceneVehicles=vehicles;vehicleSummary=data.vehicle_summary||null;plateSummary=data.recognized_plate_summary||null;$('stage-timing').textContent=`Detection wall time: ${data.detection_time_ms??'—'} ms · Vehicle detection: ${data.vehicle_detection_time_ms??'—'} ms · Angle correction: ${data.rectification_time_ms??0} ms · Crop encoding: ${data.crop_time_ms??'—'} ms. Detection includes preprocessing and postprocessing. Parallel stages overlap.${PlateExport.vehiclePlateSearchTiming(data.vehicle_plate_search)}`;$('ocr-latency').textContent=(data.read_text??wantsText())?`${Math.round(data.ocr_time_ms||0)} ms`:'Off';$('ocr-note').textContent=(data.read_text??wantsText())?'Scores are model estimates, not verified accuracy':'Plate regions retained without reading text';status(`${data.detector_model||'Plate detector'} · ${(data.read_text??wantsText())?(data.ocr_engine||'Plate reader'):'Plate localization · text reading off'} · ${(data.warnings||[]).join(' ')||'Analysis complete. Review plate crops and vehicle categories.'}${stream&&window.PlateReID?.enabled()?' · Cross-camera matching is off for live-camera frames.':''}`);
 }else{sceneVehicles=[];vehicleSummary=null;plateSummary=null;$('stage-timing').textContent='';updateTracks(candidates());tracks.forEach(t=>{t.readText=wantsText();if(!wantsText()){t.text='';t.confidence=0;}});$('ocr-latency').textContent=wantsText()?'Pending':'Off';status(`${tracks.length} plate-like candidates. Browser mode has no vehicle classifier or angle correction.`);}
 draw(vehicles);render();const elapsed=performance.now()-start;samples.push(elapsed);if(samples.length>100)samples.shift();$('latency').textContent=`${elapsed.toFixed(1)} ms`;const p95=[...samples].sort((a,b)=>a-b)[Math.ceil(samples.length*.95)-1];$('latency-note').textContent=`p95 ${p95.toFixed(1)} ms · ${samples.length} frames · ${serverMs===null?'':`server ${serverMs.toFixed(1)} ms · `} ${$('mode').value==='api'?(wantsText()?'includes network + detection + OCR':'includes network + plate/vehicle detection + crops'):'browser candidate search'} · ${elapsed<40?'within':'above'} 40 ms target`;if($('mode').value==='browser'&&wantsText()){if(recording)await recognize(token,true);else void recognize(token,!running);}return true;
 }catch(e){if(token===generation&&e.name!=='AbortError')status(`Analysis failed: ${e.message}`);return false;}finally{requestAbort=null;busy=false;controls();}}
async function tick(){if(!running)return;const token=generation,start=performance.now();if(!document.hidden&&!video.paused&&video.readyState>=2&&video.currentTime!==lastFrame){lastFrame=video.currentTime;await detect();}if(running&&token===generation)timer=setTimeout(tick,Math.max(0,40-(performance.now()-start)));}
$('media-input').addEventListener('change',async event=>{
 const file=event.target.files[0];if(!file)return;stop(true);const token=generation;url=URL.createObjectURL(file);$('source-name').textContent=file.name;
 if(file.type.startsWith('video/')||/\.(mp4|webm|mov|avi|mkv|m4v)$/i.test(file.name)){
   if($('mode').value==='api'){void PlateExport.start(file);return;}
   recording=true;source=video;video.hidden=false;video.controls=true;const controller=new AbortController();scanAbort=controller;
   try{const loaded=videoEvent('loadeddata',controller.signal);video.src=url;video.load();await loaded;if(token!==generation)return;controls();void analyseRecording();}
   catch(error){if(token===generation)status(error.message);}
 }else{
   const img=new Image();img.onload=()=>{if(token!==generation)return;source=img;imageFile=file;preview.width=img.naturalWidth;preview.height=img.naturalHeight;ctx.drawImage(img,0,0);$('empty').hidden=true;controls();void detect();};img.onerror=()=>status('Image cannot be decoded. Use JPG, PNG or WEBP.');img.src=url;
 }
});
$('camera-button').onclick=async()=>{stop(true);const token=generation;try{const incoming=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1920},height:{ideal:1080},facingMode:{ideal:'environment'}},audio:false});if(token!==generation){incoming.getTracks().forEach(t=>t.stop());return;}stream=incoming;video.srcObject=stream;await video.play();if(token!==generation)return;source=video;running=true;$('source-name').textContent='LIVE CAMERA';controls();void tick();}catch(e){if(token===generation){stop();status(`Camera unavailable: ${e.message}. Use HTTPS or localhost and allow camera access.`);}}};
$('stop-button').onclick=()=>{
 if(PlateExport.active()){void PlateExport.pause();return;}
 if(recording){generation++;running=false;scanAbort?.abort();requestAbort?.abort();video.pause();video.controls=true;controls();status('Recording analysis paused. Resume to continue from the next unanalysed frame.');}
 else{stop();status('Monitoring stopped. Last analysed frame remains visible.');}
};
$('detect-button').onclick=()=>PlateExport.active()?void PlateExport.resume():recording?void analyseRecording():void detect();
$('reset-button').onclick=()=>{stop(true);['latency','ocr-latency','vehicle-count'].forEach(id=>$(id).textContent='—');status('Session reset. Choose an input source.');};
$('mode').onchange=()=>{stop(true);$('resolution').value=$('mode').value==='api'?'1920':'640';status('Engine changed. Choose an input source.');};
$('ocr-model').onchange=()=>{samples.length=0;tracks=[];if(source&&!running&&!PlateExport.active())void detect();};
$('read-text').onchange=()=>{samples.length=0;tracks=[];controls();if(source&&!running&&!busy&&!PlateExport.active())void detect();};
$('angle-correction').onchange=()=>{samples.length=0;if(source&&!running&&!busy&&!PlateExport.active())void detect();};
$('green-filter').onchange=()=>{samples.length=0;tracks=[];if(source&&!running&&!busy&&!PlateExport.active())void detect();};
$('image-profile').onchange=()=>{samples.length=0;if(source&&!running)void detect();};
$('resolution').onchange=()=>{
 if(recording){generation++;running=false;scanAbort?.abort();requestAbort?.abort();video.pause();tracks=[];samples.length=0;controls();status('Resolution changed. Resume recording analysis to continue.');}
 else{generation++;tracks=[];samples.length=0;if(source&&!running)void detect();else if(running){clearTimeout(timer);void tick();}}
};
video.addEventListener('play',()=>{if(recording){if(running)video.pause();return;}if(source===video&&!running){running=true;controls();void tick();}});
video.addEventListener('error',()=>status('Video cannot be decoded. Try a browser-compatible MP4 or WebM.'));
const probe=setInterval(async()=>{if(window.cv?.Mat){ready=true;clearInterval(probe);$('engine-status').textContent='VISION ENGINE READY';status('Ready. Choose an image, video or camera.');controls();if(source&&!running&&$('mode').value==='browser'){if(recording)void analyseRecording();else void detect();}}},200);setTimeout(()=>{if(!ready){clearInterval(probe);$('engine-status').textContent='ENGINE UNAVAILABLE';status('Computer vision did not load. Reload or use the model API.');}},30000);window.addEventListener('pagehide',()=>{stop();workerPromise?.then(w=>w.terminate()).catch(()=>{});});
})();
