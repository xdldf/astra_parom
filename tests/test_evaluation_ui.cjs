const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
async function ui(){
 const elements=new Map(),requests=[];
 const element=()=>({value:'',hidden:false,disabled:false,style:{},parentElement:{style:{}},classList:{toggle(){}},reportValidity(){return true;}});
 const record={id:'car',version:1,created_at:'2026-10-08T10:00:00',measured_length_m:4.2,
  full_frame_photo:'car-frame.jpg',bbox:[100,50,200,100],image_size:[600,500],measurement:{warnings:[]}};
 let responder=()=>({data:{rows:[record],count:1}});
 const sandbox=vm.createContext({document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);}},
  localStorage:{getItem(){return null;},setItem(){}},fetch:async(path,options)=>{requests.push({path,options});
   const response=await responder(path,options);return {ok:response.status!==409,json:async()=>response.data};}});
 vm.runInContext(fs.readFileSync(require.resolve('../web_app/static/evaluation.js'),'utf8'),sandbox);
 await new Promise(resolve=>setImmediate(resolve));
 return {elements,requests,record,run:code=>vm.runInContext(code,sandbox),respond:fn=>responder=fn};
}
test('full-frame review keeps real length empty and a conflict preserves typed value',async()=>{
 const page=await ui();assert.equal(page.elements.get('actual').value,'');
 assert.equal(page.elements.get('measured').textContent,'4.200 м');
 assert.match(page.elements.get('photo').src,/car-frame.jpg/);
 page.elements.get('actual').value='4.8';page.respond(()=>({status:409,data:{detail:'Запись изменена'}}));
 await page.run("submit('label')");
 assert.equal(page.elements.get('actual').value,'4.8');
 assert.match(page.elements.get('message').textContent,/изменена/);
 assert.equal(page.elements.get('save').disabled,false);
 assert.equal(JSON.parse(page.requests.at(-1).options.body).actual_length_m,4.8);
});
test('skip saves a null label and advances to empty queue with export still available',async()=>{
 const page=await ui();page.respond((path,options)=>options?.method==='POST'?{data:{...page.record,evaluation:{state:'skipped'}}}:{data:{rows:[],count:0}});
 await page.run("submit('skip')");
 const payload=JSON.parse(page.requests.find(r=>r.options?.method==='POST').options.body);
 assert.equal(payload.action,'skip');assert.equal(payload.actual_length_m,null);
 assert.equal(page.elements.get('empty').hidden,false);assert.equal(page.elements.get('review').hidden,true);
 assert.equal(page.elements.get('export').hidden,false);
});
