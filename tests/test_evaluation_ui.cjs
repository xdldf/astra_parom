const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
async function ui(){
 const elements=new Map(),requests=[];
 const element=()=>({value:'',hidden:false,disabled:false,style:{},attributes:{},setAttribute(k,v){this.attributes[k]=v;},parentElement:{style:{}},classList:{toggle(){}},reportValidity(){return true;}});
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
test('outline evidence uses full-frame coordinates and keeps the original box estimate visible',async()=>{
 const page=await ui();page.elements.get('showBox').checked=true;
 page.record.measurement.outline={status:'estimated',baseline_length_m:4.4,contour:[[100,50],[300,50],[300,150]]};
 page.respond(()=>({data:{rows:[page.record],count:1}}));await page.run('load()');
 assert.equal(page.elements.get('outline').attributes.viewBox,'0 0 600 500');
 assert.equal(page.elements.get('outlinePoints').attributes.points,'100,50 300,50 300,150');
 assert.equal(page.elements.get('outline').hidden,false);
 assert.match(page.elements.get('basis').textContent,/4.400 м/);
 page.elements.get('showBox').checked=false;await page.run("$('showBox').onchange()");
 assert.equal(page.elements.get('outline').hidden,true);
});
test('validation role submits physical identity and uncertainty, and blocks catalogue evidence',async()=>{
 const page=await ui();page.elements.get('datasetRole').value='validation';
 page.elements.get('referenceSource').value='catalogue';page.elements.get('actual').value='4.7';
 const before=page.requests.length;await page.run("submit('label')");
 assert.equal(page.requests.length,before);assert.match(page.elements.get('message').textContent,/физический замер/);
 page.elements.get('referenceSource').value='physical_measurement';
 page.elements.get('physicalId').value='car-one';page.elements.get('uncertainty').value='0.01';
 page.respond((path,options)=>options?.method==='POST'?{data:{...page.record,evaluation:{state:'labeled',dataset_role:'validation'}}}:{data:{rows:[],count:0}});
 await page.run("submit('label')");
 const payload=JSON.parse(page.requests.find(r=>r.options?.method==='POST').options.body);
 assert.equal(payload.dataset_role,'validation');assert.equal(payload.physical_vehicle_id,'car-one');
 assert.equal(payload.reference_uncertainty_m,.01);assert.equal(payload.reference_source,'physical_measurement');
 assert.match(page.elements.get('message').textContent,/не используется в калибровке/);
});
