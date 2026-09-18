/* Shared by the browser and the recording scheduler regression tests. */
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.PlateVideo=api;
})(typeof window==='undefined'?globalThis:window,()=>{
  'use strict';
  function sampleTimes(duration,interval){
    if(!Number.isFinite(duration)||duration<=0)throw new Error('Recording duration is unavailable. Try an MP4 with a valid duration.');
    if(!Number.isFinite(interval)||interval<.1)throw new Error('Select a frame interval of at least 0.1 seconds.');
    const end=Math.max(0,duration-.001),times=[];
    for(let index=0;index*interval<end;index++)times.push(Number((index*interval).toFixed(6)));
    if(!times.length||end-times[times.length-1]>.001)times.push(end);
    return times;
  }
  async function scan({duration,interval,from=0,seek,analyze,active,onProgress}){
    const times=sampleTimes(duration,interval);
    for(let index=0;index<times.length;index++){
      const time=times[index];
      if(time<from-1e-6)continue;
      if(!active())return false;
      await seek(time);
      if(!active())return false;
      await analyze(time);
      if(!active())return false;
      onProgress({time,nextTime:times[index+1]??duration,index:index+1,total:times.length,complete:index===times.length-1});
    }
    return active();
  }
  function reading(track,text,confidence,status,temporal){
    track.latestText=text;
    if(!temporal){Object.assign(track,{text,confidence,formatStatus:status});return;}
    track.readings=(track.readings||[]).slice(-4);
    track.readings.push({text,confidence,status});
    const votes=new Map();
    for(const value of track.readings){
      if(value.confidence<70||value.text.length<6||value.text.length>12)continue;
      const vote=votes.get(value.text)||{score:0,count:0,confidence:0,status:value.status};
      vote.score+=value.confidence;vote.count++;vote.confidence=Math.max(vote.confidence,value.confidence);
      votes.set(value.text,vote);
    }
    // Prefer an actually observed format-matching reading over an incomplete
    // alternative. This selects OCR evidence; it never changes characters.
    const best=[...votes].sort((a,b)=>Number(b[1].status==='valid')-Number(a[1].status==='valid')||b[1].score-a[1].score)[0];
    if(best)Object.assign(track,{text:best[0],confidence:best[1].confidence,formatStatus:best[1].status,observations:best[1].count});
    else Object.assign(track,{text,confidence,formatStatus:status,observations:1});
  }
  return {sampleTimes,scan,reading};
});
