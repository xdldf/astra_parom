const {test}=require('node:test');const assert=require('node:assert/strict');const Tracks=require('../web_app/static/live-tracks.js');
test('one capture state follows overlapping detections and separates simultaneous cars',()=>{const t=new Tracks();const a=t.update([{bbox:[0,0,100,50]},{bbox:[200,0,100,50]}],0);a[0].sent=true;const b=t.update([{bbox:[5,0,100,50]},{bbox:[205,0,100,50]}],.1);assert.equal(b[0].sent,true);assert.equal(b[1].sent,false);assert.notEqual(b[0].id,b[1].id);});
test('a later passage gets a new capture state',()=>{const t=new Tracks();t.update([{bbox:[0,0,100,50]}],0)[0].sent=true;assert.equal(t.update([{bbox:[0,0,100,50]}],4)[0].sent,false);});
test('latency jump retains identity without overlap, ambiguous neighbours do not',()=>{
 const t=new Tracks(),first=t.update([{bbox:[10,100,100,60]}],0)[0];first.sent=true;
 assert.equal(t.update([{bbox:[160,102,100,62]}],1)[0].id,first.id);
 const two=t.update([{bbox:[310,100,100,60]},{bbox:[320,104,100,60]}],2);
 assert.ok(two.every(row=>row.id!==first.id));
});
test('closest real frame survives disappearance and can be flushed at video end',()=>{
 const t=new Tracks(),d={bbox:[50,100,100,60],depth:.4,status:'waiting_for_line',line_offset_px:-200};
 const a=t.update([d],0)[0];t.remember(a,{frame:1},d);
 const next={...d,bbox:[100,100,100,60],line_offset_px:-150};t.update([next],.5);t.remember(a,{frame:2},next);
 assert.equal(t.reviewsDue(.6).length,0);
 assert.equal(t.reviewsDue(.6,true)[0].review.snapshot.frame,2);
 t.update([],4);assert.equal(t.reviewsDue(4)[0].id,a.id);
 a.sent=true;assert.equal(t.reviewsDue(5).length,0);
});
test('a visible stopped car waits for its centre instead of expiring the best photo',()=>{
 const t=new Tracks(),d={bbox:[50,100,100,60],depth:.4,status:'waiting_for_line',line_offset_px:-200};
 const track=t.update([d],0)[0];t.remember(track,{frame:1},d);
 for(let time=.5;time<=5;time+=.5){t.update([d],time);assert.equal(t.reviewsDue(time).length,0);}
 assert.equal(t.reviewsDue(6).length,0);
 assert.equal(t.reviewsDue(7)[0].id,track.id);
});
