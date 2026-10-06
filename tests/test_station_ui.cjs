const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');

async function stationUI({search='',storage={}}={}){
  const elements=new Map(),requests=[];
  let clientConfig={rows_per_page:20},clientConfigVersion=1;
  function element(){return {value:'',hidden:false,children:[],parentElement:{},dataset:{},
    style:{setProperty(name,value){this[name]=String(value);}},
    classList:{values:new Set(),add(name){this.values.add(name);},remove(name){this.values.delete(name);},
      contains(name){return this.values.has(name);},toggle(name,force){const enabled=force??!this.contains(name);if(enabled)this.add(name);else this.remove(name);}},
    replaceChildren(...items){this.children=items;},append(...items){this.children.push(...items);},
    setAttribute(name,value){this[name]=value;},showModal(){this.open=true;},addEventListener(){},removeAttribute(name){delete this[name];},getAttribute(name){return this[name];}};}
  const document={hidden:false,body:element(),querySelectorAll:()=>[],addEventListener(){},createElement:element,
    getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);}};
  let responder=()=>({status:404,data:{detail:'Not configured'}});
  const sandbox=vm.createContext({document,console,URLSearchParams,FormData,structuredClone,
    localStorage:{getItem:key=>storage[key]??null,setItem(key,value){storage[key]=value;}},location:{search,origin:'http://station'},
    window:{addEventListener(){}},LiveTracks:require('../web_app/static/live-tracks.js'),setTimeout:()=>1,clearTimeout(){},setInterval(){},
    fetch:async(path,options={})=>{
      requests.push({path,options});
      let result;
      if(path.startsWith('/api/station/client-configuration?')){
        if(options.method==='POST'){clientConfig=JSON.parse(options.body);clientConfigVersion++;}
        const etag='client-config-'+clientConfigVersion;
        result=options.headers?.['If-None-Match']===etag?{status:304,etag}:{data:clientConfig,etag};
      }else result=await responder(path,options);
      const {status=200,data,etag='v1'}=result;
      return {ok:status>=200&&status<300,status,json:async()=>data,headers:{get:()=>etag}};
    }});
  vm.runInContext(fs.readFileSync(require.resolve('../web_app/static/station.js'),'utf8'),sandbox);
  await new Promise(resolve=>setImmediate(resolve));
  const initialRequests=requests.slice();
  requests.length=0;
  return {elements,requests,initialRequests,storage,document,respond:fn=>{responder=fn;},
    configureClient:config=>{clientConfig=config;clientConfigVersion++;},run:code=>vm.runInContext(code,sandbox)};
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

test('queue shows a labelled OCR number and distinguishes missing or failed readings',async()=>{
  const ui=await stationUI();
  const cases=[
    [{state:'queued',candidates:[]},'Читаем номер…',null],
    [{state:'review',candidates:[{text:'A123BC14'}],candidate_count:1},'А123ВС14',/Распознано/],
    [{state:'not_found',candidates:[]},'Номер не найден',/читаемого номера/],
    [{state:'error',candidates:[]},'Ошибка чтения номера',/повторите/],
    // The compact API only sends the first candidate, plus the actual count.
    [{state:'review',candidates:[{text:'A123BC14'}],candidate_count:2},'Несколько вариантов',/выберите номер/],
  ];
  for(const [ocr,text,note] of cases){
    const data=page();Object.assign(data.rows[0],{plate:'',front_photo:'cab.jpg',plate_ocr:ocr});
    ui.respond(()=>({data}));await ui.run('loadVehicles()');
    const cell=ui.elements.get('carsTable').children[0].children[2];
    assert.equal(cell.textContent,text);
    if(note)assert.match(cell.children[0].textContent,note);
  }
  assert.equal(ui.run('dirty'),false);
  assert.ok(ui.requests.every(r=>!r.options.method));
});

test('confirmed operator number takes priority over OCR and reports keep that number',async()=>{
  const ui=await stationUI();
  const row=page().rows[0];row.plate_ocr={state:'review',candidates:[{text:'B456EE14'}]};
  const actual=ui.run(`tableRow(${JSON.stringify(row)},true)`);
  assert.equal(actual.children[3].textContent,'А123ВС14');
  assert.equal(actual.children[3].children.length,0);
});

test('record header shows the OCR proposal without overwriting the operator field',async()=>{
  const ui=await stationUI();
  const record={...page().rows[0],plate:'',front_photo:'cab.jpg',plate_ocr:{state:'review',candidates:[{text:'A123BC14',photo:'plate.jpg'}]}};
  ui.run(`renderRecord(${JSON.stringify(record)})`);
  assert.equal(ui.elements.get('plate').textContent,'А123ВС14');
  assert.match(ui.elements.get('plate').children[0].textContent,/Распознано/);
  assert.equal(ui.elements.get('plateInput').value,'');
  assert.equal(ui.run('dirty'),false);
  ui.elements.get('ocrCandidates').children[0].children[3].onclick();
  assert.equal(ui.elements.get('plateInput').value,'А123ВС14');
  assert.equal(ui.run('dirty'),true);
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

test('video polling processes the middle crossing even when the latest frame passed the line',async()=>{
  const ui=await stationUI();
  ui.run(`streamId='clip';liveGeneration=1;liveSource={media:{id:'clip',fps:25,frames:100},profile:{measurement_line_x:300}};
    $('autoMeasure').checked=true;var captures=[];queueLive=async(snapshot,d)=>{captures.push(snapshot.frame)};`);
  const results=[340,300,260].map((center,i)=>({sequence:i+1,frame:20+2*i,detections:[{
    bbox:[center-50,150,100,75],bottom:[center,225],depth:.5,label:'car',
    length_m:center===300?5:null,at_measurement_line:center===300,status:center===300?'depth_calibrated':'waiting_for_line'}]}));
  ui.respond(path=>({data:{frame:24,result:results[2],results:path.endsWith('=3')?[]:results}}));
  await ui.run("pollStream('clip',1)");
  assert.equal(ui.run('JSON.stringify(captures)'),'[22]');
  assert.equal(ui.run('streamResultCursor'),3);
  await ui.run("pollStream('clip',1)");
  assert.ok(ui.requests.at(-1).path.endsWith('after_sequence=3'));
  assert.equal(ui.run('captures.length'),1);
});

test('skipped video crossing sends both observed boxes for server-side neighbourhood search',async()=>{
  const ui=await stationUI();
  ui.run('loadVehicles=async()=>{};renderRecord=()=>{};');
  ui.respond(path=>({data:path.endsWith('/capture-crossing')?{captured:true,record:{id:'found'}}:{id:'found'}}));
  assert.equal(await ui.run(`captureCrossing({media:{id:'clip',fps:25},profile:{detector_model:'rtdetr-x'}},30,
    {bbox:[200,150,100,75],label:'car'},{box:[300,150,100,75],time:.4})`),true);
  const request=ui.requests.find(r=>r.path.endsWith('/capture-crossing'));
  const body=JSON.parse(request.options.body);
  assert.equal(body.frame,30);assert.equal(body.before_frame,10);
  assert.deepEqual(body.before_bbox,[300,150,100,75]);
  assert.equal(body.temporal,true);assert.equal(body.source,'rtdetr-x');
  assert.ok(ui.requests.every(r=>!r.path.includes('/workbench/frame/')));
});

test('missing lengths explain camera rejection in queue and legacy record details',async()=>{
  const ui=await stationUI();const data=page();
  data.rows[0].length_m=null;data.rows[0].measurement_reason='insufficient_temporal_frames';
  ui.respond(()=>({data}));await ui.run('loadVehicles()');
  const length=ui.elements.get('carsTable').children[0].children[3];
  assert.equal(length.textContent,'Не измерена');
  assert.match(length.children[0].textContent,/Недостаточно кадров/);
  const legacy=ui.run("measurementNote({length_m:null,source:{measurement:{status:'temporal_review',temporal:{reasons:['missing_line_evidence','line_not_bracketed']}}}})");
  assert.match(legacy,/Нет пригодного кадра/);assert.match(legacy,/до и после/);
  assert.equal(ui.run("measurementNote({length_m:4.5,measurement_reason:'outside_calibration'})"),'');
  assert.match(ui.run("measurementNote({length_m:null,measurement_reason:'anchor_outside_calibration'})"),/вне области калибровки/);
});

test('operator model badge uses the running IP server profile and clears stale state',async()=>{
  const ui=await stationUI();ui.run("ipMode=true;page='operator';importedProfile={detector_model:'yolo26m'};");
  ui.respond(()=>({data:{running:true,id:'camera',detector:{model:'rtdetr-x',imgsz:1280},auto_measure:true}}));
  await ui.run('ipPoll()');
  assert.equal(ui.elements.get('activeDetector').textContent,'RT-DETR X · 1280 px');
  ui.run('calibrationLabel()'); // A late video-profile load must not overwrite IP status.
  assert.equal(ui.elements.get('activeDetector').textContent,'RT-DETR X · 1280 px');
  ui.respond(()=>({data:{running:false,detector:{model:'yolo26l',imgsz:640}}}));
  await ui.run('ipPoll()');assert.equal(ui.elements.get('activeDetector').textContent,'YOLO26 L · 640 px');
  assert.match(ui.elements.get('activeDetectorContext').textContent,/остановлены/);
  ui.respond(()=>({status:503,data:{detail:'offline'}}));await ui.run('ipPoll()');
  assert.equal(ui.elements.get('activeDetector').textContent,'Не определена');
  ui.run("ipMode=false;importedProfile={detector_model:'rtdetr-l',detector_imgsz:640};calibrationLabel();");
  assert.equal(ui.elements.get('activeDetector').textContent,'RT-DETR L · 640 px');
});

test('running video badge uses server detector rather than old browser settings',async()=>{
  const ui=await stationUI();
  ui.run("liveSource={media:{id:'clip',frames:1000,fps:25},profile:{detector_model:'yolo26m'}};streamId='video';liveGeneration=1;");
  ui.respond(()=>({data:{frame:10,results:[],detector:{model:'rtdetr-x',imgsz:640}}}));
  await ui.run("pollStream('video',1)");
  assert.equal(ui.elements.get('activeDetector').textContent,'RT-DETR X · 640 px');
  ui.run('calibrationLabel()');assert.equal(ui.elements.get('activeDetector').textContent,'RT-DETR X · 640 px');
});

test('approximate saved lengths retain their numeric value and visible review reason',async()=>{
  const ui=await stationUI();const data=page();
  Object.assign(data.rows[0],{length_m:4.5,measurement_approximate:1,measurement_reason:'unstable_temporal_length'});
  ui.respond(()=>({data}));await ui.run('loadVehicles()');
  const length=ui.elements.get('carsTable').children[0].children[3];
  assert.match(length.textContent,/^≈ /);assert.match(length.textContent,/4,5/);
  assert.match(length.children[0].textContent,/Приблизительно/);assert.match(length.children[0].textContent,/расходится/);
});

test('operator detector selection survives polling and applies to running IP source',async()=>{
 const ui=await stationUI();ui.run("ipMode=true;page='operator';showDetector({model:'yolo26m',imgsz:640},'server');");
 ui.elements.get('operatorDetectorModel').value='rtdetr-x';ui.elements.get('operatorDetectorModel').onchange();
 ui.run("showDetector({model:'yolo26m',imgsz:640},'server')");
 assert.equal(ui.elements.get('operatorDetectorModel').value,'rtdetr-x');
 ui.respond((path)=>({data:path.endsWith('/detector')?{detector:{model:'rtdetr-x',imgsz:640},running:true,restarted:true}:{running:true,id:'new-session',detector:{model:'rtdetr-x',imgsz:640}}}));
 await ui.elements.get('applyDetector').onclick();
 const req=ui.requests.find(r=>r.path==='/api/ip/detector');
 assert.deepEqual(JSON.parse(req.options.body),{detector_model:'rtdetr-x',detector_imgsz:640});
 assert.match(ui.elements.get('activeDetector').textContent,/RT-DETR X/);
 assert.match(ui.elements.get('operatorDetectorStatus').textContent,/переподключены/);
 assert.equal(ui.run('detectorEditDirty'),false);
 assert.ok(ui.requests.every(r=>r.path!=='/api/ip/stop'));
});

test('failed model change preserves the running badge and keeps the requested selection for retry',async()=>{
 const ui=await stationUI();ui.run("ipMode=true;showDetector({model:'yolo26m',imgsz:640},'server');");
 ui.elements.get('operatorDetectorModel').value='rtdetr-x';ui.elements.get('operatorDetectorModel').onchange();
 ui.respond(()=>({status:503,data:{detail:'Weights unavailable'}}));
 await ui.elements.get('applyDetector').onclick();
 assert.match(ui.elements.get('activeDetector').textContent,/YOLO26 M/);
 assert.match(ui.elements.get('operatorDetectorStatus').textContent,/Weights unavailable/);
 assert.equal(ui.elements.get('operatorDetectorModel').value,'rtdetr-x');
 assert.equal(ui.run('detectorApplying'),false);
});

test('server video calibration wins over browser cache after reload, even without a saved camera pair',async()=>{
 const ui=await stationUI();
 ui.run("liveSource={media:{id:'local-video',image_size:[600,500]},profile:{detector_model:'yolo26m',image_size:[600,500]}};importedProfile=null;");
 ui.respond(path=>path==='/api/stream/configuration'?{status:404,data:{}}:{data:{detector_model:'rtdetr-x',detector_imgsz:640,image_size:[600,500]}});
 await ui.run('restoreVideoSettings()');
 assert.equal(ui.run('liveSource.profile.detector_model'),'rtdetr-x');
 assert.match(ui.elements.get('activeDetector').textContent,/RT-DETR X/);
});

test('late saved-profile response cannot replace a model in a running video',async()=>{
 const ui=await stationUI();ui.run("liveSource={media:{id:'video',image_size:[600,500]},profile:{detector_model:'rtdetr-x'}};importedProfile=null;streamId='running';");
 ui.respond(path=>path==='/api/stream/configuration'?{status:404,data:{}}:{data:{detector_model:'yolo26m',image_size:[600,500]}});
 await ui.run('restoreVideoSettings()');
 assert.equal(ui.run('liveSource.profile.detector_model'),'rtdetr-x');
 assert.equal(ui.run('streamId'),'running');
});

test('video model preflight failure leaves existing video session and profile intact',async()=>{
 const ui=await stationUI();
 ui.run("ipMode=false;streamId='old';liveSource={media:{id:'video'},profile:{detector_model:'yolo26m'}};showDetector({model:'yolo26m',imgsz:640},'server');");
 ui.elements.get('operatorDetectorModel').value='rtdetr-x';ui.elements.get('operatorDetectorModel').onchange();
 ui.respond(()=>({status:503,data:{detail:'CUDA unavailable'}}));
 await ui.elements.get('applyDetector').onclick();
 assert.equal(ui.run('streamId'),'old');assert.equal(ui.run('liveSource.profile.detector_model'),'yolo26m');
 assert.equal(ui.requests.length,1);assert.equal(ui.requests[0].path,'/api/workbench/detector/check');
 assert.match(ui.elements.get('operatorDetectorStatus').textContent,/CUDA unavailable/);
});

test('client lists unconfirmed and paid cars with side photos and explicit confirmation tags',async()=>{
  const ui=await stationUI();
  const pending={...page().rows[0],plate:'',side_photo:'side.jpg',front_photo:'front.jpg',
    plate_ocr:{state:'review',candidates:[{text:'A123BC14',photo:'plate.jpg'}]}};
  const paid={...page().rows[0],id:'paid',status:'Оплачен',length_m:17.16,category:'truck_capacity'};
  ui.respond(()=>({data:{...page(),rows:[pending,paid],count:2,has_more:false,tariffs_enabled:true}}));
  await ui.run('syncClient()');
  assert.match(ui.requests.find(r=>r.path.includes('/vehicles?')).path,/\/vehicles\?limit=20&offset=0$/);
  const [first,second]=ui.elements.get('clientRows').children;
  assert.equal(first.children.length,4);
  assert.equal(first.children[0].children[0].textContent,'А123ВС14');
  assert.equal(first.children[0].children[1].textContent,'Не подтвержден');
  assert.equal(first.children[1].children.length,1);
  assert.equal(first.children[1].children[0].src,'/api/station/photos/side.jpg?v=1');
  assert.equal(first.children[3].children.length,1);
  assert.equal(second.children[0].children[1].textContent,'Подтвержден');
  assert.equal(second.children[2].textContent,'17,16 м');
  assert.equal(second.children[3].children.length,1);
  assert.equal(ui.elements.get('clientEmpty').hidden,true);
});

test('client handles missing or failed photos, ambiguous OCR, missing lengths and rejected prices',async()=>{
  const ui=await stationUI();
  const missing={...page().rows[0],plate:'',length_m:null,front_photo:'front-only.jpg',
    plate_ocr:{state:'review',candidate_count:2,candidates:[{text:'A123BC14'}]},tariff:{amount_rub:null}};
  let row=ui.run(`clientRow(${JSON.stringify(missing)})`);
  assert.equal(row.children[0].children[0].textContent,'Несколько вариантов');
  assert.equal(row.children[1].children.length,0);
  assert.equal(row.children[1].textContent,'Фото отсутствует');
  assert.equal(row.children[2].textContent,'Не измерена');
  assert.equal(row.children[3].children[0].textContent,'Уточняется');
  row=ui.run(`clientRow(${JSON.stringify({...missing,status:'Отклонён',side_photo:'broken.jpg',tariff:{amount_rub:1390}})})`);
  row.children[1].children[0].onerror();
  assert.equal(row.children[1].textContent,'Фото недоступно');
  assert.equal(row.children[0].children[1].textContent,'Не подтвержден');
  assert.equal(row.children[3].children[0].textContent,'—');
});

test('client polling keeps unchanged rows, freezes older pages and returns to live arrivals',async()=>{
  const ui=await stationUI();
  ui.respond((path,options)=>options.headers?.['If-None-Match']?{status:304}:{data:page(Number(new URL(path,'http://station').searchParams.get('offset')))});
  await ui.run('syncClient()');const row=ui.elements.get('clientRows').children[0];
  await ui.run('syncClient()');
  assert.equal(ui.elements.get('clientRows').children[0],row);
  assert.equal(ui.requests.at(-1).options.headers['If-None-Match'],'v1');
  await ui.elements.get('clientNext').onclick();
  assert.match(ui.requests.at(-1).path,/offset=20&snapshot=501$/);
  assert.equal(ui.elements.get('clientRange').textContent,'21–21 из 501');
  assert.equal(ui.elements.get('clientLatest').disabled,false);
  await ui.elements.get('clientLatest').onclick();
  assert.match(ui.requests.at(-1).path,/offset=0$/);
  assert.equal(ui.elements.get('clientLatest').disabled,true);
});

test('client ignores stale page responses and marks old data when connection fails',async()=>{
  const ui=await stationUI();let finish;
  ui.respond(()=>new Promise(resolve=>{finish=resolve;}));
  const older=ui.run('syncClient()');
  await new Promise(resolve=>setImmediate(resolve));
  ui.respond(()=>({data:{...page(),rows:[],count:0,has_more:false,tariffs_enabled:false}}));
  await ui.run('syncClient()');finish({data:page()});await older;
  assert.equal(ui.elements.get('clientRows').children.length,0);
  assert.equal(ui.elements.get('clientEmpty').hidden,false);
  assert.equal(ui.elements.get('clientData').hidden,true);
  assert.equal(ui.run('tariffsEnabled'),false);
  ui.respond(()=>{throw Error('offline');});
  await assert.rejects(ui.run('syncClient()'),/offline/);
  assert.equal(ui.elements.get('clientConnection').hidden,false);
  assert.match(ui.elements.get('clientConnection').textContent,/последние полученные/);
  ui.respond(()=>({data:page()}));await ui.run('syncClient()');
  assert.equal(ui.elements.get('clientConnection').hidden,true);
});

test('standalone client starts even without catalog or GPU health endpoints',async()=>{
  const ui=await stationUI({search:'?page=client&station=desk-1'});
  assert.equal(ui.run('page'),'client');
  assert.ok(ui.document.body.classList.contains('client-only'));
  const stationRequests=ui.initialRequests.filter(r=>r.path.startsWith('/api/station/'));
  assert.equal(stationRequests.length,2);
  assert.match(stationRequests[0].path,/\/client-configuration\?station_id=desk-1$/);
  assert.match(stationRequests[1].path,/\/api\/station\/vehicles\?limit=20&offset=0$/);
  ui.respond(()=>({data:page()}));await ui.run('start()');
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(ui.requests.every(r=>r.path.startsWith('/api/station/vehicles?')||r.path.startsWith('/api/station/client-configuration?')));
});

test('operator price visibility persists locally without changing tariffs, quotes or unsaved edits',async()=>{
  const ui=await stationUI();
  ui.run("current={id:'editing',status:'Подтвержден',tariff:{amount_rub:1390}};dirty=true;showQuote(current.tariff);");
  ui.elements.get('plateInput').value='МОИ ПРАВКИ';
  const confirm=ui.elements.get('confirm');
  assert.equal(confirm.children.at(-1).className,'operator-price');
  const requests=ui.requests.length;
  ui.elements.get('showOperatorPrices').checked=false;ui.elements.get('showOperatorPrices').onchange();
  assert.ok(ui.document.body.classList.contains('operator-prices-hidden'));
  assert.equal(ui.storage.ferryOperatorPrices,'false');
  assert.equal(ui.run('tariffsEnabled'),true);
  assert.equal(ui.run('dirty'),true);
  assert.equal(ui.elements.get('plateInput').value,'МОИ ПРАВКИ');
  assert.equal(ui.elements.get('confirm').disabled,false);
  assert.equal(ui.requests.length,requests);
  const reopened=await stationUI({storage:ui.storage});
  assert.equal(reopened.elements.get('showOperatorPrices').checked,false);
  assert.ok(reopened.document.body.classList.contains('operator-prices-hidden'));
  reopened.elements.get('showOperatorPrices').checked=true;reopened.elements.get('showOperatorPrices').onchange();
  assert.ok(!reopened.document.body.classList.contains('operator-prices-hidden'));
});

test('client settings save to the selected desk and reset the preview to newest cars',async()=>{
  const ui=await stationUI();ui.respond(path=>({data:page(Number(new URL(path,'http://station').searchParams.get('offset')))}));
  await ui.run('syncClient()');await ui.elements.get('clientNext').onclick();
  ui.elements.get('stationId').value='cashier-2';
  ui.elements.get('clientRowsPerPage').value='10';await ui.elements.get('clientRowsPerPage').onchange();
  const saved=ui.requests.find(r=>r.options.method==='POST');
  assert.equal(saved.path,'/api/station/client-configuration?station_id=cashier-2');
  assert.deepEqual(JSON.parse(saved.options.body),{rows_per_page:10});
  assert.match(ui.requests.at(-1).path,/limit=10&offset=0$/);
  assert.equal(ui.run('clientPageSize'),10);
  assert.equal(ui.elements.get('client').style['--client-page-size'],'10');
});

test('open client receives changed display settings while staying on the latest page',async()=>{
  const ui=await stationUI({search:'?page=client&station=desk-1'});ui.respond(()=>({data:page()}));
  await ui.run('syncClient()');
  assert.equal(ui.run('clientPageSize'),20);
  assert.ok(ui.elements.get('client').classList.contains('client-dense'));
  ui.configureClient({rows_per_page:5});await ui.run('syncClient()');
  assert.equal(ui.run('clientPageSize'),5);
  assert.match(ui.requests.at(-1).path,/limit=5&offset=0$/);
  assert.ok(!ui.elements.get('client').classList.contains('client-dense'));
  assert.ok(ui.requests.every(r=>!r.options.method));
});
