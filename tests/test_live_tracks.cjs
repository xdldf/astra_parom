const {test}=require('node:test');const assert=require('node:assert/strict');const Tracks=require('../web_app/static/live-tracks.js');
test('one capture state follows overlapping detections and separates simultaneous cars',()=>{const t=new Tracks();const a=t.update([{bbox:[0,0,100,50]},{bbox:[200,0,100,50]}],0);a[0].sent=true;const b=t.update([{bbox:[5,0,100,50]},{bbox:[205,0,100,50]}],.1);assert.equal(b[0].sent,true);assert.equal(b[1].sent,false);assert.notEqual(b[0].id,b[1].id);});
test('a later passage gets a new capture state',()=>{const t=new Tracks();t.update([{bbox:[0,0,100,50]}],0)[0].sent=true;t.update([],.1);assert.equal(t.update([{bbox:[0,0,100,50]}],4)[0].sent,false);});
test('latency jump retains identity without overlap, ambiguous neighbours do not',()=>{
 const t=new Tracks(),first=t.update([{bbox:[10,100,100,60]}],0)[0];first.sent=true;
 assert.equal(t.update([{bbox:[160,102,100,62]}],1)[0].id,first.id);
 const two=t.update([{bbox:[310,100,100,60]},{bbox:[320,104,100,60]}],2);
 assert.ok(two.every(row=>row.id!==first.id));
});
test('a counted stopped car survives slow inference',()=>{
 const t=new Tracks(),d={bbox:[250,150,100,75],passage_id:'one'};
 const track=t.update([d],0)[0];track.sent=true;
 for(const time of [3,6,12])assert.equal(t.update([d],time)[0].sent,true);
});
test('server passage identity survives a shrinking tail and separates following cars',()=>{
 const t=new Tracks();t.update([{bbox:[220,150,200,75],passage_id:'one'}],0)[0].sent=true;
 const rows=t.update([{bbox:[295,150,25,75],passage_id:'one'},{bbox:[250,150,100,75],passage_id:'two'}],3);
 assert.equal(rows[0].sent,true);assert.equal(rows[1].sent,false);
 assert.notEqual(rows[0].id,rows[1].id);
});
test('server captured state protects a browser that missed the earlier capture',()=>{
 const t=new Tracks();
 assert.equal(t.update([{bbox:[250,150,100,75],passage_id:'one',passage_captured:true}],10)[0].sent,true);
});
