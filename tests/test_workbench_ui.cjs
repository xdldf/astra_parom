// Exercise the actual frontend event handlers without a browser or model download.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function studio() {
  const elements = new Map();
  const strokes = [];
  const context = new Proxy({}, {get: (_,key) => (...args) => {
    if (key === 'measureText') return {width:100};
    strokes.push([key,...args]);
  }});
  function element() {
    return {value:'0',textContent:'',className:'',hidden:false,checked:false,style:{},classList:{add(){},remove(){}},
      width:600,height:500,clientWidth:600,children:[],
      append(...items){this.children.push(...items);},
      replaceChildren(){this.children=[];},
      getContext(){return context;}, setPointerCapture(){},
      getBoundingClientRect(){return {left:0,top:0,width:600,height:500};}};
  }
  const document = {getElementById(id){
    if(!elements.has(id))elements.set(id,element());
    return elements.get(id);
  },createElement:element};
  const sandbox = vm.createContext({document,console,structuredClone,FormData,
    setTimeout,clearTimeout,Image:class {set src(_){this.width=600;this.height=500;this.onload();}},
    fetch:async (_url,options)=>{
      const body=JSON.parse(options.body);
      if(_url.endsWith('validate-profile')||_url.endsWith('approved-references')) {
        if(body.polygon.length<4) return {ok:false,json:async()=>({detail:'Invalid road'})};
        return {ok:true,json:async()=>body};
      }
      return {ok:true,json:async()=>({image:'test',detections:[],scale:null})};
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../web_app/static/app.js'),'utf8'),sandbox);
  vm.runInContext(`media={id:'test',frames:2,image_size:[600,500]}; picture={};
    profile={image_size:[600,500],lens:{...defaults},polygon:[],references:[]};
    draft=[[20,200],[580,200],[580,450],[20,450]]; mode='road';`,sandbox);
  return {elements,strokes,run:code=>vm.runInContext(code,sandbox)};
}

test('known-length slider and Detect preserve an unfinished four-point road',async()=>{
  const ui=studio();
  ui.elements.get('length').value='5.2';
  ui.elements.get('length').oninput();
  assert.equal(ui.run('draft.length'),4);
  await ui.elements.get('detect').onclick();
  assert.equal(ui.run('profile.polygon.length'),4);
  assert.equal(ui.run('mode'),'select');
  assert.equal(ui.run('draft.length'),0);
  ui.run('draw()');
  assert.ok(ui.strokes.some(s=>s[0]==='lineTo'&&s[1]===580&&s[2]===450));
});

test('setting a reference preserves the saved road',async()=>{
  const ui=studio();
  await ui.run('finishPendingRoad()');
  const before=ui.run('JSON.stringify(profile.polygon)');
  ui.run(`detections=[{bbox:[100,150,100,75],label:'car'}];selected=0;`);
  ui.elements.get('length').value='5';
  await ui.elements.get('addRef').onclick();
  assert.equal(ui.run('JSON.stringify(profile.polygon)'),before);
  assert.equal(ui.run('profile.references[0].length_m'),5);
});

test('repeated Add reference updates one observation without changing its weight',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  ui.run('refresh=async()=>{};detections=[{bbox:[100,150,100,75],label:"car"}];selected=0;');
  ui.elements.get('length').value='5';await ui.elements.get('addRef').onclick();
  ui.elements.get('length').value='5.1';await ui.elements.get('addRef').onclick();
  assert.equal(ui.run('profile.references.length'),1);
  assert.equal(ui.run('profile.references[0].length_m'),5.1);
});

test('calibration inconsistency is visible and never labelled validated',()=>{
  const ui=studio();
  ui.run('showCalibrationDiagnostics({diagnostics:{reference_max_abs_error_m:.162,references_within_target:3,unique_reference_count:8,duplicates_ignored:8}})');
  assert.match(ui.elements.get('status').textContent,/16.2 см/);
  assert.match(ui.elements.get('status').textContent,/не подтверждена/);
  assert.equal(ui.elements.get('status').className,'error');
});

test('raised-barrier geometry failure is visible separately from detector stability',()=>{
  const ui=studio();
  ui.run('showCalibrationDiagnostics({diagnostics:{survey:true,ruler_check_error_m:.097,ground_check_error_m:.438,target_tolerance_m:.1}})');
  assert.match(ui.elements.get('status').textContent,/43.8 см/);
  assert.match(ui.elements.get('status').textContent,/не подтверждена/);
  assert.equal(ui.elements.get('status').className,'error');
  ui.run('profile.survey_calibration={};refresh=async()=>{};invalidateLens();');
  assert.equal(ui.run('profile.survey_calibration'),null);
});

test('model selection survives export profile and discards stale detector boxes',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  ui.run('profile.references=[{bbox:[100,150,100,75],length_m:5}];detections=[{bbox:[100,150,100,75]}];selected=0;detectorUI();');
  assert.equal(ui.elements.get('detectorModel').value,'yolo26n');
  ui.elements.get('detectorModel').value='yolo26m';
  ui.elements.get('detectorSize').value='1280';
  await ui.elements.get('detectorModel').onchange();
  assert.equal(ui.run('profile.detector_model'),'yolo26m');
  assert.equal(ui.run('profile.detector_imgsz'),1280);
  assert.equal(ui.run('profile.references.length'),1);
  assert.equal(ui.run('detections.length'),0);
  assert.match(ui.elements.get('status').textContent,/проверьте калибровку/);
});

test('an incomplete road is retained instead of hidden on detection',async()=>{
  const ui=studio();
  ui.run('draft.pop()');
  await ui.elements.get('detect').onclick();
  assert.equal(ui.run('mode'),'road');
  assert.equal(ui.run('draft.length'),3);
});

test('line placement and dragging preserve road and references',async()=>{
  const ui=studio();
  await ui.elements.get('placeLine').onclick();
  assert.equal(ui.run('profile.polygon.length'),4);
  assert.equal(ui.run('mode'),'line');
  // Stub refresh to keep this pointer test independent of frame rendering.
  ui.run('refresh=async()=>{}');
  const canvas=ui.elements.get('canvas');
  canvas.onpointerdown({clientX:300,clientY:100,pointerId:1});
  assert.equal(ui.run('profile.measurement_line_x'),300);
  canvas.onpointerdown({clientX:300,clientY:100,pointerId:1});
  canvas.onpointermove({clientX:340,clientY:100});
  await canvas.onpointerup();
  assert.equal(ui.run('profile.measurement_line_x'),340);
  assert.equal(ui.run('profile.polygon.length'),4);
});

test('play continues through bbox center crossings to the end',async()=>{
  const ui=studio();
  await ui.run('finishPendingRoad()');
  ui.run(`profile.measurement_line_x=150;media.frames=4;media.fps=25;
    refresh=async detect=>{if(!detect)throw Error('Detection was skipped');
      detections=[{bbox:[100,150,100,75],label:'car',depth:.1,
        at_measurement_line:true,status:'needs_reference'}];};`);
  await ui.elements.get('play').onclick();
  assert.equal(Number(ui.elements.get('timeline').value),3);
  assert.equal(ui.run('playing'),false);
  assert.equal(ui.elements.get('play').textContent,'▶ Пуск');
  assert.doesNotMatch((ui.elements.get('status')?.textContent||''),/Paused: bbox center/);
});

test('disabling a line also finishes and preserves a valid road draft',async()=>{
  const ui=studio();
  ui.run('profile.measurement_line_x=150');
  await ui.elements.get('removeLine').onclick();
  assert.equal(ui.run('profile.measurement_line_x'),null);
  assert.equal(ui.run('profile.polygon.length'),4);
});


test('metre marks on a half-size preview are stored at original image coordinates',async()=>{
  const ui=studio();
  await ui.run('finishPendingRoad()');
  ui.run('refresh=async()=>{}');
  ui.run("$('rulerStep').value='1'");
  const canvas=ui.elements.get('canvas');
  canvas.getBoundingClientRect=()=>({left:30,top:40,width:300,height:250});
  await ui.elements.get('drawRuler').onclick();
  for(const clientX of [80,130,155,180,230])canvas.onpointerdown({clientX,clientY:190,pointerId:1});
  await ui.elements.get('finishRuler').onclick();
  assert.deepEqual(JSON.parse(ui.run('JSON.stringify(profile.metric_rulers[0])')),
    {points:[[100,300],[200,300],[250,300],[300,300],[400,300]],step_m:1});
  assert.equal(ui.run('mode'),'select');
});

test('lens changes invalidate rulers drawn in the old coordinate system',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  ui.run('refresh=async()=>{};profile.metric_rulers=[{points:[[100,300],[200,300],[300,300]],step_m:1}];invalidateLens()');
  assert.equal(ui.run('profile.metric_rulers.length'),0);
});

const continuedProfile={version:1,image_size:[600,500],lens:{k1:-.2,tilt_deg:2},
  polygon:[[20,200],[580,200],[580,450],[20,450]],
  references:[{bbox:[100,150,100,75],length_m:5,frame:12}],
  metric_rulers:[{points:[[100,300],[200,300],[300,300]],step_m:1}],
  measurement_line_x:350,line_tolerance_px:8};

test('JSON can be imported before media, then retained on repeated camera frames',async()=>{
  const ui=studio();
  ui.run('media=null;profile=null;picture=null;draft=[];mode="select";');
  const input={files:[{name:'existing.json',text:async()=>JSON.stringify({profile:continuedProfile})}],value:'existing.json'};
  await ui.elements.get('import').onchange({target:input});
  assert.equal(ui.run('profile.references[0].length_m'),5);
  assert.equal(ui.run('profile.lens.tilt_deg'),2);
  assert.equal(ui.elements.get('lineTolerance').value,8);
  const before=ui.run('JSON.stringify(profile)');
  ui.run("api=async path=>({id:path,frames:1,fps:0,image_size:[600,500],name:'Camera frame',source:'ip_camera',captured_at:'2026-09-21T00:00:00Z'});refresh=async()=>{};");
  await ui.elements.get('cameraFrame').onclick();
  await ui.elements.get('cameraFrame').onclick();
  assert.equal(ui.run('JSON.stringify(profile)'),before);
  assert.equal(ui.elements.get('calibrationTarget').value,'ip');
  assert.equal(ui.elements.get('play').disabled,true);
});

test('resolution mismatch never silently clears loaded calibration',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  const before=ui.run('JSON.stringify(profile)');
  ui.run("api=async()=>({id:'new',image_size:[1200,1000],frames:1,name:'Different resolution'});refresh=async()=>{};");
  await ui.elements.get('cameraFrame').onclick();
  assert.equal(ui.run('JSON.stringify(profile)'),before);
  assert.equal(ui.run('media.id'),'test');
  assert.equal(ui.elements.get('resolutionMismatch').hidden,false);
  await ui.elements.get('useNewImage').onclick();
  assert.equal(ui.run('media.id'),'new');
  assert.equal(ui.run('profile.image_size[0]'),1200);
  assert.equal(ui.run('profile.references.length'),0);
});

test('saved calibration loads without an image and uses the selected station source',async()=>{
  const ui=studio();ui.run('media=null;profile=null;draft=[];mode="select";');
  ui.run("$('calibrationTarget').value='ip'");
  ui.run(`stationProfileRequest=async method=>{if(method!=='GET')throw Error('Unexpected save');return ${JSON.stringify(continuedProfile)}};`);
  await ui.elements.get('loadSaved').onclick();
  assert.equal(ui.run('profile.metric_rulers[0].points.length'),3);
  assert.match(ui.elements.get('profileInfo').textContent,/Сохранённая калибровка IP/);
});

test('imported reference lengths and ruler spacing remain editable',async()=>{
  const ui=studio();ui.run('media=null;profile=null;draft=[];mode="select";');
  await ui.run(`loadProfile(${JSON.stringify(continuedProfile)},'existing.json')`);
  ui.elements.get('references').children[0].children[1].value='6.2';
  await ui.elements.get('references').children[0].children[1].onchange();
  assert.equal(ui.run('profile.references[0].length_m'),6.2);
  ui.elements.get('rulers').children[0].children[1].value='2';
  await ui.elements.get('rulers').children[0].children[1].onchange();
  assert.equal(ui.run('profile.metric_rulers[0].step_m'),2);
});

test('camera failure keeps the current image and complete calibration',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  const before=ui.run('JSON.stringify(profile)');
  ui.run("api=async()=>{throw Error('Camera offline')}");
  await ui.elements.get('cameraFrame').onclick();
  assert.equal(ui.run('JSON.stringify(profile)'),before);
  assert.equal(ui.run('media.id'),'test');
  assert.match(ui.elements.get('status').textContent,/Camera offline/);
});

test('imported ruler marks can be dragged in original pixel coordinates',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  await ui.run(`loadProfile(${JSON.stringify(continuedProfile)},'existing.json')`);
  ui.run('picture={};refresh=async()=>{}');
  const canvas=ui.elements.get('canvas');
  canvas.onpointerdown({clientX:200,clientY:300,pointerId:1});
  canvas.onpointermove({clientX:210,clientY:300});
  await canvas.onpointerup();
  assert.equal(ui.run('profile.metric_rulers[0].points[1][0]'),210);
});

test('apply saves edited calibration without discarding it on a running-camera conflict',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  await ui.run(`loadProfile(${JSON.stringify(continuedProfile)},'existing.json')`);
  ui.run("stationProfileRequest=async()=>{throw Error('Сначала остановите камеры')}");
  await ui.elements.get('applyCalibration').onclick();
  assert.match(ui.elements.get('status').textContent,/Сначала остановите камеры/);
  assert.equal(ui.run('profile.references[0].length_m'),5);
  ui.run("stationProfileRequest=async(method,body)=>{if(method!=='POST'||body.measurement_line_x!==350)throw Error('Wrong profile');return {saved:true}};");
  await ui.elements.get('applyCalibration').onclick();
  assert.match(ui.elements.get('status').textContent,/Калибровка сохранена/);
});


test('position inspector draws the local road cross-section instead of a global horizontal line',()=>{
  const ui=studio();
  ui.run(`detections=[{bbox:[200,150,200,150],label:'car',depth:.5,road_cross_section:{far:[300,150],near:[300,450]}}];selected=0;draw();inspect();`);
  assert.ok(ui.strokes.some(s=>s[0]==='moveTo'&&s[1]===300&&s[2]===150));
  assert.ok(ui.strokes.some(s=>s[0]==='lineTo'&&s[1]===300&&s[2]===450));
  assert.equal(ui.elements.get('depth').textContent,'50.0% дороги');
});


test('RT-DETR selection survives profile import and clears stale YOLO boxes',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  ui.run('profile.detector_model="rtdetr-x";profile.detector_imgsz=640;detectorUI();');
  assert.equal(ui.elements.get('detectorModel').value,'rtdetr-x');
  assert.equal(ui.elements.get('detect').textContent,'Найти · RT-DETR X');
  ui.run('detections=[{bbox:[100,150,100,75],source:"yolo26m"}];selected=0;');
  ui.elements.get('detectorModel').value='rtdetr-l';
  ui.elements.get('detectorSize').value='640';
  await ui.elements.get('detectorModel').onchange();
  assert.equal(ui.run('profile.detector_model'),'rtdetr-l');
  assert.equal(ui.run('detections.length'),0);
});

test('RT-DETR can be selected before opening media and reaches the first detection request',async()=>{
  const ui=studio();
  ui.run('media=null;profile=null;picture=null;draft=[];mode="select";detectorUI();');
  ui.elements.get('detectorModel').value='rtdetr-x';
  await ui.elements.get('detectorModel').onchange();
  ui.elements.get('detectorSize').value='1280';
  await ui.elements.get('detectorSize').onchange();
  assert.equal(ui.elements.get('detectorModel').value,'rtdetr-x');
  assert.equal(ui.run('profile'),null);
  assert.match(ui.elements.get('detect').textContent,/RT-DETR X/);
  ui.run('adoptMedia({id:"new",name:"new.mp4",frames:20,image_size:[600,500]});');
  assert.equal(ui.run('profile.detector_model'),'rtdetr-x');
  assert.equal(ui.run('profile.detector_imgsz'),1280);
  ui.run('var requestBody;fetch=async(url,options)=>{requestBody=JSON.parse(options.body);return {ok:true,json:async()=>({image:"test",detections:[],scale:null})}};');
  await ui.elements.get('detect').onclick();
  assert.equal(ui.run('requestBody.profile.detector_model'),'rtdetr-x');
  assert.equal(ui.run('requestBody.profile.detector_imgsz'),1280);
});

test('detector controls show processing state and recover after a failed request',async()=>{
  const ui=studio();
  ui.run('var rejectFrame;api=()=>new Promise((resolve,reject)=>{rejectFrame=reject});');
  const request=ui.run('refresh(true)');
  assert.equal(ui.elements.get('detectorModel').disabled,true);
  assert.equal(ui.elements.get('detectorSize').disabled,true);
  assert.match(ui.elements.get('detectorStatus').textContent,/Дождитесь/);
  ui.run('rejectFrame(Error("Model unavailable"))');
  await request;
  assert.equal(ui.elements.get('detectorModel').disabled,false);
  assert.equal(ui.elements.get('detectorSize').disabled,false);
  ui.elements.get('detectorModel').value='rtdetr-l';
  await ui.elements.get('detectorModel').onchange();
  assert.equal(ui.run('profile.detector_model'),'rtdetr-l');
});

test('imported detector settings take priority over an unbound selection',async()=>{
  const ui=studio();
  ui.run('media=null;profile=null;picture=null;draft=[];mode="select";detectorUI();');
  ui.elements.get('detectorModel').value='rtdetr-x';
  await ui.elements.get('detectorModel').onchange();
  await ui.run(`loadProfile(${JSON.stringify({...continuedProfile,detector_model:'yolo26l',detector_imgsz:640})},'saved.json')`);
  assert.equal(ui.elements.get('detectorModel').value,'yolo26l');
  assert.equal(ui.run('profile.detector_model'),'yolo26l');
});

test('saved detector switches without loading any image and survives a fresh load',async()=>{
  const ui=studio();ui.run('media=null;profile=null;picture=null;draft=[];mode="select";');
  ui.run(`var stored=${JSON.stringify({...continuedProfile,detector_model:'yolo26m',detector_imgsz:640})};
    $('calibrationTarget').value='ip';stationProfileRequest=async(method,body)=>{
      if(method==='POST'){stored=structuredClone(body);return {profile:stored,restarted:true};}
      return structuredClone(stored);
    };`);
  await ui.elements.get('loadSaved').onclick();
  assert.equal(ui.elements.get('saveState').textContent,'Настройка сохранена');
  ui.elements.get('detectorModel').value='rtdetr-x';await ui.elements.get('detectorModel').onchange();
  assert.match(ui.elements.get('saveState').textContent,/неприменённые/);
  assert.match(ui.elements.get('serverDetector').textContent,/YOLO26 M/);
  await ui.elements.get('applyCalibration').onclick();
  assert.equal(ui.run('stored.detector_model'),'rtdetr-x');
  assert.deepEqual(JSON.parse(ui.run('JSON.stringify(stored.polygon)')),continuedProfile.polygon);
  assert.match(ui.elements.get('status').textContent,/переподключены/);
  assert.match(ui.elements.get('serverDetector').textContent,/RT-DETR X/);
  ui.run('profile=null');await ui.elements.get('loadSaved').onclick();
  assert.equal(ui.elements.get('detectorModel').value,'rtdetr-x');
});

test('undo restores road and metric calibration invalidated by a lens edit',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  await ui.run(`loadProfile(${JSON.stringify(continuedProfile)},'saved.json');refresh=async()=>{};`);
  const before=ui.run('JSON.stringify(profile)');
  ui.elements.get('k1').value=-.3;ui.elements.get('k1').oninput();
  assert.equal(ui.run('profile.polygon.length'),0);
  await ui.elements.get('undoChange').onclick();
  assert.equal(ui.run('JSON.stringify(profile)'),before);
});

test('changing source keeps independent unsaved drafts and does not copy IP geometry to video',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');
  ui.run(`currentTarget='ip';$('calibrationTarget').value='ip';profile.detector_model='rtdetr-x';
    stationProfileRequest=async()=>(${JSON.stringify({...continuedProfile,detector_model:'yolo26m'})});refresh=async()=>{};`);
  ui.elements.get('calibrationTarget').value='operator';await ui.elements.get('calibrationTarget').onchange();
  assert.equal(ui.run('profile.detector_model'),'yolo26m');
  ui.elements.get('calibrationTarget').value='ip';await ui.elements.get('calibrationTarget').onchange();
  assert.equal(ui.run('profile.detector_model'),'rtdetr-x');
  assert.equal(ui.run('media.id'),'test');
});

test('zoom selected before the image arrives is applied when the corrected frame finishes',async()=>{
  const ui=studio();await ui.run('finishPendingRoad()');ui.run('picture=null;');
  ui.elements.get('viewZoom').value='2';ui.elements.get('viewZoom').onchange();
  assert.equal(ui.elements.get('canvas').style.width,'');
  await ui.run('refresh()');
  assert.equal(ui.elements.get('canvas').style.width,'1200px');
  assert.equal(ui.run('profile.image_size[0]'),600);
});
