/* Shared client: run scripts/sync-client.ps1 after editing. */
(() => {
'use strict';
const $=id=>document.getElementById(id), preview=$('preview'), ctx=preview.getContext('2d'), video=$('video');
const frame=document.createElement('canvas'), fc=frame.getContext('2d',{willReadFrequently:true});
let imageFile=null;
let source=null,stream=null,url=null,generation=0,running=false,ready=false,busy=false,ocrBusy=false,workerPromise=null;
let sceneVehicles=[],tracks=[],nextId=1,lastFrame=-1,timer=null;const samples=[];
let recording=false,recordingNext=0,recordingDone=false,scanAbort=null,requestAbort=null;
const recordingReadings=new Map();
$('mode').value=new URLSearchParams(location.search).get('engine')==='browser'?'browser':'api';
$('resolution').value=$('mode').value==='api'?'1920':'640';
const status=text=>{$('status').textContent=text;};
const valid=text=>/^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$/.test(text);
function overlap(a,b){const area=Math.max(0,Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y));return area/Math.max(1,a.width*a.height+b.width*b.height-area);}
function controls(){if(window.PlateExport?.active()){PlateExport.controls();return;}$('detect-button').disabled=!source||running||busy||($('mode').value==='browser'&&!ready);$('detect-button').textContent=recording?(recordingDone?'Analyse recording again':'Resume recording analysis'):'Analyse image';$('stop-button').textContent=recording?'Pause':'Stop';$('stop-button').disabled=!running;$('video-interval').disabled=running;}
function stop(clear=false){window.PlateExport?.reset();generation++;running=false;scanAbort?.abort();requestAbort?.abort();recording=false;recordingNext=0;recordingDone=false;recordingReadings.clear();$('recording-progress').hidden=true;$('recording-log').hidden=true;clearTimeout(timer);video.pause();stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.removeAttribute('src');video.load();video.hidden=true;if(url)URL.revokeObjectURL(url);url=null;source=null;imageFile=null;tracks=[];samples.length=0;lastFrame=-1;if(clear){sceneVehicles=[];$('media-input').value='';ctx.clearRect(0,0,preview.width,preview.height);$('empty').hidden=false;$('source-name').textContent='NO SOURCE';render();}controls();}

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
    if(!track.text||track.text.length<6)continue;
    const item=recordingReadings.get(track.id)||{text:track.text,first:time,last:time,count:0,confidence:0,crop:track.crop,formatStatus:track.formatStatus};
    item.last=time;item.count++;if(track.text!==item.text||track.confidence>=item.confidence){item.text=track.text;item.confidence=track.confidence;item.crop=track.crop;item.formatStatus=track.formatStatus;}
    recordingReadings.set(track.id,item);
  }
  $('recording-log').hidden=false;
  const rows=$('recording-rows');rows.replaceChildren();
  for(const item of recordingReadings.values()){
    const row=rows.insertRow();row.insertCell().textContent=`${timestamp(item.first)} – ${timestamp(item.last)}`;
    const img=document.createElement('img');img.src=item.crop.toDataURL('image/jpeg',.9);img.alt=`Recorded plate ${item.text}`;row.insertCell().append(img);
    row.insertCell().textContent=item.text;row.insertCell().textContent=item.count;
    row.insertCell().textContent=item.confidence>=70&&item.formatStatus==='valid'?'Format matched · verify crop':'Uncertain · review crop';
  }
  if(!recordingReadings.size){const cell=rows.insertRow().insertCell();cell.colSpan=6;cell.className='empty-row';cell.textContent='No plate text has been read in the analysed frames.';}
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
    if(completed&&token===generation){recordingDone=true;status(`Recording complete. ${recordingReadings.size} plate tracks with text. Review uncertain readings against their crops.`);}
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
function draw(vehicles=[]){preview.width=frame.width;preview.height=frame.height;ctx.drawImage(frame,0,0);const box=(b,label,color)=>{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(b.x,b.y,b.width,b.height);const font=Math.max(14,Math.round(frame.width/85)),height=font+10;ctx.font=`600 ${font}px Arial`;const width=ctx.measureText(label).width+16,y=Math.max(height,b.y);ctx.fillStyle=color;ctx.fillRect(b.x,y-height,width,height);ctx.fillStyle='#102234';ctx.fillText(label,b.x+8,y-6);};vehicles.forEach((v,i)=>box(v.bounding_box,`${v.label} ${i+1}`,'#7ec8ff'));tracks.forEach(t=>box(t.box,t.text||`Unverified ${t.id}`,t.text&&t.confidence>=70&&valid(t.text)?'#58dfa1':'#e9bc73'));$('empty').hidden=true;}
function cropPreview(t){const c=document.createElement('canvas');c.width=t.crop.width;c.height=t.crop.height;c.getContext('2d').drawImage(t.crop,0,0);c.setAttribute('role','img');c.setAttribute('aria-label',`Plate region ${t.id}`);return c;}
function render(){renderVehicles();$('plate-count').textContent=tracks.length;const rows=$('result-rows');rows.replaceChildren();if(!tracks.length){const cell=rows.insertRow().insertCell();cell.colSpan=6;cell.className='empty-row';cell.textContent='No plate candidates in the current frame.';return;}tracks.forEach(t=>{const row=rows.insertRow();row.insertCell().textContent=`Plate #${t.id}`;const img=cropPreview(t);row.insertCell().append(img);const text=document.createElement('strong');text.textContent=t.text||(t.attempts?'Unreadable':'Pending OCR');row.insertCell().append(text);row.insertCell().textContent=Number.isFinite(t.detectionConfidence)?`${Math.round(t.detectionConfidence)}%`:'—';row.insertCell().textContent=t.confidence?`${Math.round(t.confidence)}%`:'—';row.insertCell().textContent=t.text?(valid(t.text)&&t.confidence>=70?'Format matched · verify crop':'Uncertain · review required'):'Candidate · unverified';});}
function renderVehicles(){
 const panel=$('vehicle-cards');panel.replaceChildren();
 const readable=tracks.filter(t=>t.text).length;
 $('vehicle-count').textContent=readable;
 $('vehicle-note').textContent='Regions with recognised text';
 $('coverage').textContent=tracks.length?`${readable} text readings · ${tracks.length} plate regions`:'Awaiting plate detections';
 if(!tracks.length){const empty=document.createElement('div');empty.className='vehicle-empty';empty.textContent='Plate crops appear here after analysing an image or video.';panel.append(empty);return;}
 tracks.forEach(t=>{const card=document.createElement('article');card.className='vehicle-card';const img=cropPreview(t);const copy=document.createElement('div');const title=document.createElement('span');title.className='vehicle-title';title.textContent=`Plate region ${t.id}`;const reading=document.createElement('strong');reading.textContent=t.text||(t.attempts?'Text unreadable':'Reading text…');const detail=document.createElement('small');detail.textContent=t.text?`${Number.isFinite(t.detectionConfidence)?Math.round(t.detectionConfidence)+"%":"—"} detection · ${Math.round(t.confidence)}% OCR · verify crop`:'Unverified region · review crop';copy.append(title,reading,detail);card.append(img,copy);panel.append(card);});
}
async function getWorker(){if(!workerPromise)workerPromise=(async()=>{if(!window.Tesseract)throw new Error('OCR library unavailable. Check your network and reload.');const worker=await Tesseract.createWorker('eng');await worker.setParameters({tessedit_char_whitelist:'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',tessedit_pageseg_mode:'6'});return worker;})().catch(e=>{workerPromise=null;throw e;});return workerPromise;}
async function recognize(token,all=false){if(ocrBusy)return;ocrBusy=true;try{const worker=await getWorker();if(token!==generation)return;const pending=tracks.filter(t=>!t.lastOcr||performance.now()-t.lastOcr>1200).sort((a,b)=>a.lastOcr-b.lastOcr).slice(0,all?20:2);for(const t of pending){if(token!==generation)break;const start=performance.now(),result=await worker.recognize(t.crop);if(token!==generation)break;t.lastOcr=performance.now();t.attempts++;const text=result.data.text.toUpperCase().replace(/[^A-Z0-9]/g,''),confidence=result.data.confidence;if(text.length>=6&&text.length<=12&&confidence>=45){t.votes[text]=(t.votes[text]||0)+confidence;const best=Object.keys(t.votes).sort((a,b)=>t.votes[b]-t.votes[a])[0];if(best===text){t.text=text;t.confidence=confidence;}}$('ocr-latency').textContent=`${Math.round(performance.now()-start)} ms`;render();}}catch(e){if(token===generation)status(e.message);}finally{ocrBusy=false;if(token!==generation&&source&&!running&&$('mode').value==='browser')void recognize(generation,true);}}
async function detect(){if(busy||!source||($('mode').value==='browser'&&!ready))return false;const token=generation;busy=true;controls();const start=performance.now();try{const w=source.videoWidth||source.naturalWidth,h=source.videoHeight||source.naturalHeight;if(!w||!h)return;const scale=Math.min(1,Number($('resolution').value)/w);frame.width=Math.round(w*scale);frame.height=Math.round(h*scale);fc.drawImage(source,0,0,frame.width,frame.height);let vehicles=[],serverMs=null;
 if($('mode').value==='api'){const blob=(!running&&imageFile&&scale===1)?imageFile:await new Promise(resolve=>frame.toBlob(resolve,'image/jpeg',.95));const form=new FormData();form.append('image',blob,imageFile?.name||'frame.jpg');form.append('profile',$('image-profile').value);requestAbort=new AbortController();const response=await requestModel(`${$('api-url').value.replace(/\/$/,'')}/api/detect/${running?'frame':'image'}`,form,requestAbort.signal);const data=await response.json();serverMs=data.processing_time_ms;if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Model API rejected the frame.');if(token!==generation)return;updateTracks(data.detections.map(d=>d.bounding_box));data.detections.forEach((d,i)=>{PlateVideo.reading(tracks[i],d.normalized_text,d.ocr_confidence*100,d.format_status,recording||!!stream);tracks[i].detectionConfidence=d.detection_confidence*100;tracks[i].attempts=1;tracks[i].vehicleId=d.vehicle_id;});vehicles=data.vehicles||[];sceneVehicles=vehicles;$('ocr-latency').textContent=`${Math.round(data.ocr_time_ms||0)} ms`;status((data.warnings||[]).join(' ')||'Model analysis complete. Review readings below.');
 }else{sceneVehicles=[];updateTracks(candidates());status(`${tracks.length} plate candidates. OCR runs independently; review readings against crops.`);}
 draw(vehicles);render();const elapsed=performance.now()-start;samples.push(elapsed);if(samples.length>100)samples.shift();$('latency').textContent=`${Math.round(elapsed)} ms`;const p95=[...samples].sort((a,b)=>a-b)[Math.ceil(samples.length*.95)-1];$('latency-note').textContent=`p95 ${Math.round(p95)} ms · ${samples.length} frames · ${serverMs===null?'':`server ${Math.round(serverMs)} ms · `} ${$('mode').value==='api'?'includes network + OCR':'excludes OCR'} · ${elapsed<40?'within':'above'} 40 ms target`;if($('mode').value==='browser'){if(recording)await recognize(token,true);else void recognize(token,!running);}return true;
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
$('image-profile').onchange=()=>{samples.length=0;if(source&&!running)void detect();};
$('resolution').onchange=()=>{
 if(recording){generation++;running=false;scanAbort?.abort();requestAbort?.abort();video.pause();tracks=[];samples.length=0;controls();status('Resolution changed. Resume recording analysis to continue.');}
 else{generation++;tracks=[];samples.length=0;if(source&&!running)void detect();else if(running){clearTimeout(timer);void tick();}}
};
video.addEventListener('play',()=>{if(recording){if(running)video.pause();return;}if(source===video&&!running){running=true;controls();void tick();}});
video.addEventListener('error',()=>status('Video cannot be decoded. Try a browser-compatible MP4 or WebM.'));
const probe=setInterval(async()=>{if(window.cv?.Mat){ready=true;clearInterval(probe);$('engine-status').textContent='VISION ENGINE READY';status('Ready. Choose an image, video or camera.');controls();if(source&&!running&&$('mode').value==='browser'){if(recording)void analyseRecording();else void detect();}}},200);setTimeout(()=>{if(!ready){clearInterval(probe);$('engine-status').textContent='ENGINE UNAVAILABLE';status('Computer vision did not load. Reload or use the model API.');}},30000);window.addEventListener('pagehide',()=>{stop();workerPromise?.then(w=>w.terminate()).catch(()=>{});});
})();
