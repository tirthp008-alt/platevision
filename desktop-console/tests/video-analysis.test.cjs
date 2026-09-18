const {test}=require('node:test');
const assert=require('node:assert/strict');
const {sampleTimes,scan,reading}=require('../video-analysis.js');

test('samples start, regular intervals and final frame',()=>{
  assert.deepEqual(sampleTimes(1.2,.5),[0,.5,1,1.199]);
  assert.throws(()=>sampleTimes(Infinity,.5));
  assert.throws(()=>sampleTimes(3,0));
});
test('slow analysis never skips a selected sample or overlaps requests',async()=>{
  const events=[];let inFlight=0;
  const complete=await scan({duration:1.2,interval:.5,active:()=>true,
    seek:async time=>events.push(['seek',time]),
    analyze:async time=>{assert.equal(++inFlight,1);await new Promise(resolve=>setTimeout(resolve,10));events.push(['analyze',time]);inFlight--;},
    onProgress:value=>events.push(['progress',value.time])});
  assert.equal(complete,true);
  assert.deepEqual(events.filter(e=>e[0]==='analyze').map(e=>e[1]),[0,.5,1,1.199]);
  assert.equal(events.length,12);
});
test('pause after seeking prevents inference and resume starts at pending sample',async()=>{
  let active=true;let analyzed=0;
  assert.equal(await scan({duration:2,interval:.5,active:()=>active,seek:async()=>{active=false;},analyze:async()=>analyzed++,onProgress:()=>{}}),false);
  assert.equal(analyzed,0);
  const times=[];
  await scan({duration:2,interval:.5,from:1,active:()=>true,seek:async()=>{},analyze:async time=>times.push(time),onProgress:()=>{}});
  assert.deepEqual(times,[1,1.5,1.999]);
});
test('failed inference stops recording without marking samples complete',async()=>{
  let progress=0;
  await assert.rejects(scan({duration:2,interval:.5,active:()=>true,seek:async()=>{},analyze:async()=>{throw new Error('API unavailable');},onProgress:()=>progress++}),/API unavailable/);
  assert.equal(progress,0);
});
test('temporal voting keeps repeated text through a transient bad read, then expires',()=>{
  const track={};
  reading(track,'MH12AB1234',95,'valid',true);
  reading(track,'MH12AB1234',93,'valid',true);
  reading(track,'MH12AB1239',96,'valid',true);
  assert.equal(track.text,'MH12AB1234');
  for(let i=0;i<5;i++)reading(track,'MH12AB1239',96,'valid',true);
  assert.equal(track.text,'MH12AB1239');
  reading(track,'',0,'uncertain',false);
  assert.equal(track.text,'');
});
test('format matching chooses observed OCR text without correcting characters',()=>{
  const track={};
  reading(track,'OD02BG1396',92,'valid',true);
  reading(track,'0D02BG1396',98,'possible',true);
  assert.equal(track.text,'OD02BG1396');
  const unknown={};reading(unknown,'0D02BG1396',98,'possible',true);
  assert.equal(unknown.text,'0D02BG1396');
});
