const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');

async function stationUI(){
  const elements=new Map(),requests=[];
  function element(){return {value:'',hidden:false,children:[],parentElement:{},dataset:{},
    classList:{add(){},remove(){},toggle(){}},
    replaceChildren(...items){this.children=items;},append(...items){this.children.push(...items);},
    setAttribute(name,value){this[name]=value;},showModal(){this.open=true;},addEventListener(){},removeAttribute(name){delete this[name];},getAttribute(name){return this[name];}};}
  const document={hidden:false,body:element(),querySelectorAll:()=>[],addEventListener(){},createElement:element,
    getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);}};
  let responder=()=>({status:404,data:{detail:'Not configured'}});
  const sandbox=vm.createContext({document,console,URLSearchParams,FormData,structuredClone,
    localStorage:{getItem:()=>null,setItem(){}},location:{search:'',origin:'http://station'},
    window:{addEventListener(){}},LiveTracks:class {},setTimeout:()=>1,clearTimeout(){},setInterval(){},
    fetch:async(path,options={})=>{
      requests.push({path,options});const {status=200,data,etag='v1'}=await responder(path,options);
      return {ok:status>=200&&status<300,status,json:async()=>data,headers:{get:()=>etag}};
    }});
  vm.runInContext(fs.readFileSync(require.resolve('../web_app/static/station.js'),'utf8'),sandbox);
  await new Promise(resolve=>setImmediate(resolve));
  requests.length=0;
  return {elements,requests,respond:fn=>{responder=fn;},run:code=>vm.runInContext(code,sandbox)};
}

function page(offset=0){return {rows:[{id:'car-'+offset,version:1,created_at:'2026-09-18T12:00:00',
  plate:'А123ВС14',category:'car',length_m:4,tariff:{amount_rub:1390},status:'Требует проверки'}],
  count:501,offset,snapshot:501,has_more:offset<500};}

test('polling unchanged queue preserves DOM and sends ETag',async()=>{
  const ui=await stationUI();
  ui.respond((_,options)=>options.headers?.['If-None-Match']?{status:304}:{data:page()});
  await ui.run('loadVehicles()');const row=ui.elements.get('carsTable').children[0];
  await ui.run('loadVehicles()');
  assert.equal(ui.elements.get('carsTable').children[0],row);
  assert.equal(ui.requests[1].options.headers['If-None-Match'],'v1');
  assert.match(ui.requests[0].path,/limit=250/);
});

test('older queue pages freeze insertion boundary and filters reset paging',async()=>{
  const ui=await stationUI();
  ui.respond(path=>({data:page(Number(new URL(path,'http://station').searchParams.get('offset')))}));
  await ui.run('loadVehicles()');await ui.elements.get('queueNext').onclick();
  assert.match(ui.requests.at(-1).path,/offset=250&snapshot=501/);
  ui.elements.get('filterPlate').value='А123';ui.elements.get('filterPlate').oninput();
  assert.equal(ui.run('queueOffset'),0);assert.equal(ui.run('queueSnapshot'),null);
});

test('source switch leaves shared IP cameras running',async()=>{
  const ui=await stationUI();ui.run('ipMode=true;ipRunning=true;');
  ui.elements.get('sourceMode').value='video';await ui.elements.get('sourceMode').onchange();
  assert.equal(ui.run('ipMode'),false);
  assert.ok(ui.requests.every(r=>r.path!=='/api/ip/stop'));
});

test('concurrent update warns without overwriting dirty operator fields',async()=>{
  const ui=await stationUI();
  ui.run("current={id:'car-0',version:1};dirty=true;");
  ui.elements.get('plateInput').value='МОЯ ПРАВКА';
  ui.respond(path=>({data:path.includes('/vehicles?')?page():{id:'car-0',version:2}}));
  await ui.run('loadVehicles()');
  assert.equal(ui.run('current.version'),1);
  assert.equal(ui.elements.get('plateInput').value,'МОЯ ПРАВКА');
  assert.equal(ui.elements.get('recordConflict').hidden,false);
});

test('customer URL and submitted edits carry the selected operator desk',async()=>{
  const ui=await stationUI();
  ui.elements.get('stationId').value='cashier-2';ui.elements.get('stationId').onchange();
  assert.equal(ui.elements.get('openClient').href,'/?page=client&station=cashier-2');
  ui.elements.get('categoryInput').value='truck_capacity';ui.elements.get('capacityInput').value='25';
  const fields=ui.run('fields()');assert.equal(fields.station_id,'cashier-2');assert.equal(fields.load_capacity_t,25);
});


test('row thumbnails open original photos without selecting a row or losing edits',async()=>{
  const ui=await stationUI();
  ui.run("current={id:'editing',version:1};dirty=true;");
  const data=page();data.rows[0].side_photo='side.jpg';data.rows[0].front_photo='front.jpg';
  ui.respond(path=>({data:path.includes('/vehicles?')?data:{id:'editing',version:1}}));
  await ui.run('loadVehicles()');
  const photos=ui.elements.get('carsTable').children[0].children[0];
  assert.equal(photos.children.length,2);
  const button=photos.children[1],img=button.children[0];
  assert.equal(img.loading,'lazy');assert.match(img.src,/thumbnail=true/);
  let stopped=false;button.onclick({stopPropagation(){stopped=true;}});
  assert.ok(stopped);assert.equal(ui.run('current.id'),'editing');assert.equal(ui.run('dirty'),true);
  assert.equal(ui.elements.get('expandedMeasurement').src,'/api/station/photos/front.jpg?v=1');
  assert.equal(ui.elements.get('measurementDialog').open,true);
});

test('earlier cab thumbnail explains timing and category choices only edit the form',async()=>{
  const ui=await stationUI();
  const data=page();data.rows[0].front_photo='cab.jpg';data.rows[0].front_photo_offset_seconds=-8;
  ui.respond(()=>({data}));
  await ui.run('loadVehicles()');
  ui.elements.get('carsTable').children[0].children[0].children[0].onclick({stopPropagation(){}});
  assert.match(ui.elements.get('measurementCaption').textContent,/8.0 с до синхронного/);
  ui.run("current={status:'Требует проверки'};categories={road_train:'Тягач с полуприцепом'};showQuote({amount_rub:null,warnings:['Выберите тип'],category_options:['road_train']});");
  const count=ui.requests.length;
  ui.elements.get('warningText').children.at(-1).onclick();
  assert.equal(ui.elements.get('categoryInput').value,'road_train');
  assert.equal(ui.run('dirty'),true);
  assert.equal(ui.requests.length,count);
});

test('turning tariffs off keeps unsaved edits and permits measurement confirmation',async()=>{
  const ui=await stationUI();
  ui.run("current={id:'editing',status:'Подтвержден',tariff:{amount_rub:9000,warnings:['Tariff warning'],category_options:['road_train']}};dirty=true;mode='manual';");
  ui.elements.get('plateInput').value='МОИ ПРАВКИ';
  ui.elements.get('categoryInput').value='truck';
  ui.elements.get('manualTariff').value='';
  ui.run('applyTariffMode(false)');
  assert.equal(ui.elements.get('plateInput').value,'МОИ ПРАВКИ');
  assert.equal(ui.run('dirty'),true);
  assert.equal(ui.elements.get('tariffsEnabled').checked,false);
  assert.equal(ui.elements.get('confirm').disabled,false);
  assert.equal(ui.elements.get('confirm').textContent,'✓ Подтвердить измерение');
  assert.equal(ui.elements.get('paid').disabled,true);
  assert.equal(ui.run('fields().manual_rub'),null);
  assert.equal(ui.elements.get('warningText').children.length,0);
});

test('tariff switch saves server setting and another operator receives it through queue polling',async()=>{
  const first=await stationUI();
  first.respond((path,options)=>({data:JSON.parse(options.body)}));
  first.elements.get('tariffsEnabled').checked=false;
  await first.elements.get('tariffsEnabled').onchange();
  assert.equal(first.requests[0].path,'/api/station/configuration');
  assert.equal(first.run('tariffsEnabled'),false);
  const second=await stationUI();
  second.respond(()=>({data:{...page(),tariffs_enabled:false}}));
  await second.run('loadVehicles()');
  assert.equal(second.run('tariffsEnabled'),false);
});

test('only approved cars expose reference verification and estimates are not prefilled',async()=>{
  const ui=await stationUI();
  ui.run("renderCalibrationReference({status:'Требует проверки',length_m:4.5,source:{bbox:[1,2,3,4]},side_photo:'side.jpg'})");
  assert.equal(ui.elements.get('verifyReference').disabled,true);
  assert.equal(ui.elements.get('referenceLength').value,'');
  ui.run("renderCalibrationReference({status:'Подтвержден',length_m:4.5,source:{bbox:[1,2,3,4]},side_photo:'side.jpg'})");
  assert.equal(ui.elements.get('verifyReference').disabled,false);
  assert.equal(ui.elements.get('referenceLength').value,'');
  assert.equal(ui.elements.get('referenceVerified').checked,false);
  ui.run("current={id:'old',version:3};dirty=false;");
  await assert.rejects(ui.run('saveCalibrationReference(true)'),/проверку/);
  assert.equal(ui.requests.length,0);
});

test('reference submission sends the separately verified length and actual operator',async()=>{
  const ui=await stationUI();
  ui.run("current={id:'old',version:3};dirty=false;renderRecord=()=>{};loadVehicles=async()=>{};");
  ui.run("$('referenceLength').value='6.25';$('referenceVerified').checked=true;$('actor').value='Анна';");
  ui.respond(()=>({data:{}}));
  await ui.run('saveCalibrationReference(true)');
  const request=ui.requests.at(-1);
  assert.equal(request.path,'/api/station/vehicles/old/calibration-reference');
  assert.deepEqual(JSON.parse(request.options.body),{version:3,actor:'Анна',enabled:true,actual_length_m:6.25,verified:true});
});
