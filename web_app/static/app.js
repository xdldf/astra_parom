const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
let media=null, profile=null, picture=null, detections=[], selected=-1, mode='select', draft=[], drag=null, busy=false, playing=false, revision=0, timer;
let profileName='Новая калибровка', pendingMedia=null, sessionBusy=false;
const defaults={k1:0,k2:0,focal:1,zoom:1,cx:.5,cy:.5,tilt_deg:0};
const controls=[['k1','Radial correction',-.8,.8,.005],['k2','Edge correction',-.5,.5,.005],['tilt_deg','Tilt · left / right (degrees)',-30,30,.1],['focal','Focal scale',.4,2,.01],['zoom','Zoom / crop',.5,2,.01],['cx','Lens center · horizontal',.2,.8,.005],['cy','Lens center · vertical',.2,.8,.005]];
function status(text,error=false){$('status').textContent=text;$('status').className=error?'error':'';}
async function api(path,body){let r=await fetch('/api/workbench'+path,{method:'POST',headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body instanceof FormData?body:JSON.stringify(body)});let data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
function ready(){if(!media){status('Откройте кадр с камеры, изображение или видео. Загруженная калибровка сохранится.',true);return false;}if(busy){status('Wait for the current frame to finish.');return false;}return true;}
function setMode(value){mode=value;$('mode').textContent={road:'Click road boundary points',box:'Drag around the complete car',select:'Select a car · drag road dots or measurement line',line:'Click to place the measurement line',ruler:'Click consecutive measured road marks, then Finish ruler'}[value];draw();}
function lensUI(){for(const [id] of controls){$(id).value=profile?.lens[id]??defaults[id];$(id+'Value').textContent=Number($(id).value).toFixed(3);}}
for(const [id,label,min,max,step] of controls){let l=document.createElement('label');l.className='sliderlabel';l.innerHTML=`${label}<output id="${id}Value"></output><input type="range" id="${id}" min="${min}" max="${max}" step="${step}">`;$('lensControls').append(l);$(id).oninput=()=>{if(sessionBusy){lensUI();return;}if(!media)return;profile.lens[id]=Number($(id).value);$(id+'Value').textContent=Number($(id).value).toFixed(3);invalidateLens();};}
lensUI();
function invalidateLens(){playing=false;profile.polygon=[];profile.references=[];profile.metric_rulers=[];profile.measurement_line_x=null;draft=[];detections=[];selected=-1;revision++;setMode('select');renderReferences();inspect();clearTimeout(timer);timer=setTimeout(()=>refresh(false),180);}
async function refresh(detect=false){if(!media)return;if(busy){clearTimeout(timer);timer=setTimeout(()=>refresh(detect),200);return;}busy=true;const rev=revision;const frame=Number($('timeline').value);$('detect').disabled=true;try{status(detect?'Running YOLO26n on the corrected frame… First use downloads the model.':'Updating corrected frame…');const data=await api('/frame/'+media.id,{profile,frame,detect,confidence:Number($('confidence').value),boxes:detect?[]:detections.map(d=>d.bbox)});const img=new Image();await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=reject;img.src='data:image/jpeg;base64,'+data.image;});if(rev!==revision)return;picture=img;const previous=detections;detections=data.detections.map((d,i)=>detect?{...d,source:'yolo26n'}:{...d,label:previous[i]?.label??d.label,confidence:previous[i]?.confidence??null,source:previous[i]?.source??'manual'});selected=Math.min(selected,detections.length-1);if(selected<0&&detections.length)selected=0;canvas.width=img.width;canvas.height=img.height;$('empty').hidden=true;$('frameLabel').textContent=media.frames===1?`Снимок · ${profile.image_size.join(' × ')} px`:`Frame ${frame} / ${media.frames-1}`;draw();inspect();renderReferences();status(data.scale?.status==='ruler_calibrated'?'Metric rulers active. Length integrates each marked interval; outside coverage is not measured.':data.scale?.status==='depth_calibrated'?'Depth calibration active. Select a car to inspect the scale.':data.scale?'One reference: constant scale only. Add another car at a different depth.':detect?`Found ${detections.length} vehicles. Select one and set its known length.`:'Draw the road, then add known-length reference cars.');}catch(e){status(e.message,true);playing=false;}finally{busy=false;$('detect').disabled=false;}}
function point(e){let r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height-1,(e.clientY-r.top)*canvas.height/r.height))];}
function draw(){if(!picture)return;ctx.drawImage(picture,0,0);const s=canvas.width/1000;ctx.lineWidth=2*s;ctx.font=`${13*s}px system-ui`;if($('grid').checked){ctx.strokeStyle='#ffffff35';ctx.lineWidth=s;for(let n=1;n<10;n++){ctx.beginPath();ctx.moveTo(0,n*canvas.height/10);ctx.lineTo(canvas.width,n*canvas.height/10);ctx.stroke();ctx.beginPath();ctx.moveTo(n*canvas.width/10,0);ctx.lineTo(n*canvas.width/10,canvas.height);ctx.stroke();}}let p=profile.polygon;if(p.length){ctx.beginPath();p.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle='#65e6aa20';ctx.fill();ctx.strokeStyle='#80e6bd';ctx.lineWidth=2*s;ctx.stroke();p.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,5*s,0,Math.PI*2);ctx.fillStyle='#80e6bd';ctx.fill();ctx.fillText(i+1,x+8*s,y-8*s);});}drawDraftAndLine(s);drawRulers(s);detections.forEach((d,i)=>{let [x,y,w,h]=d.bbox;ctx.strokeStyle=i===selected?'#ffe084':'#78c8ff';ctx.lineWidth=(i===selected?3:1.5)*s;ctx.strokeRect(x,y,w,h);ctx.beginPath();ctx.moveTo(x+w/2,y);ctx.lineTo(x+w/2,y+h);ctx.stroke();ctx.beginPath();ctx.arc(x+w/2,y+h/2,4*s,0,Math.PI*2);ctx.fillStyle=d.at_measurement_line?'#80e6bd':ctx.strokeStyle;ctx.fill();ctx.beginPath();ctx.moveTo(x,y+h);ctx.lineTo(x+w,y+h);ctx.stroke();ctx.beginPath();ctx.arc(x+w/2,y+h,5*s,0,Math.PI*2);ctx.fillStyle=ctx.strokeStyle;ctx.fill();let text=`${i+1} · ${d.label} ${d.confidence!=null?Math.round(d.confidence*100)+'% ':''}${d.length_m!=null?'≈ '+d.length_m.toFixed(2)+' m':''}`;let tw=ctx.measureText(text).width;ctx.fillStyle='#09151ee8';ctx.fillRect(x,Math.max(0,y-24*s),tw+12*s,24*s);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(text,x+6*s,Math.max(16*s,y-7*s));if(i===selected&&d.depth!=null){ctx.save();ctx.setLineDash([6*s,5*s]);ctx.beginPath();ctx.moveTo(0,y+h);ctx.lineTo(canvas.width,y+h);ctx.strokeStyle='#ffe08480';ctx.stroke();ctx.restore();}});if(drag?.type==='box'){let [x,y]=drag.start,[ex,ey]=drag.end;ctx.strokeStyle='#ffe084';ctx.strokeRect(x,y,ex-x,ey-y);}}
canvas.onpointerdown=e=>{if(!ready()||!picture)return;playing=false;let p=point(e);canvas.setPointerCapture(e.pointerId);if(mode==='road'||mode==='ruler'){draft.push(p);draw();return;}if(mode==='line'){profile.measurement_line_x=p[0];setMode('select');refresh();return;}if(mode==='box'){drag={type:'box',start:p,end:p};return;}let near=profile.polygon.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<12*canvas.width/canvas.clientWidth);if(near>=0){drag={type:'vertex',index:near,old:structuredClone(profile)};return;}for(let r=0;r<(profile.metric_rulers||[]).length;r++){const index=profile.metric_rulers[r].points.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<12*canvas.width/canvas.clientWidth);if(index>=0){drag={type:'ruler-point',ruler:r,index,old:structuredClone(profile)};return;}}if(profile.measurement_line_x!=null&&Math.abs(p[0]-profile.measurement_line_x)<10*canvas.width/canvas.clientWidth){drag={type:'line',old:structuredClone(profile)};return;}selected=detections.findIndex(d=>p[0]>=d.bbox[0]&&p[0]<=d.bbox[0]+d.bbox[2]&&p[1]>=d.bbox[1]&&p[1]<=d.bbox[1]+d.bbox[3]);draw();inspect();};
canvas.onpointermove=e=>{if(!drag)return;let p=point(e);if(drag.type==='box')drag.end=p;else if(drag.type==='line')profile.measurement_line_x=p[0];else if(drag.type==='ruler-point')profile.metric_rulers[drag.ruler].points[drag.index]=p;else profile.polygon[drag.index]=p;draw();};
canvas.onpointerup=async()=>{if(!drag)return;let d=drag;drag=null;if(d.type==='box'){let [x,y]=d.start,[ex,ey]=d.end;let b=[Math.min(x,ex),Math.min(y,ey),Math.abs(ex-x),Math.abs(ey-y)];if(b[2]>3&&b[3]>3){detections.push({bbox:b,label:'manual car',source:'manual'});selected=detections.length-1;setMode('select');await refresh();}}else{try{await api('/validate-profile',profile);await refresh();}catch(e){profile=d.old;status(e.message,true);draw();}}};
canvas.onpointercancel=()=>{if(drag?.old)profile=drag.old;drag=null;draw();};
function inspect(){updateLineLabel();const d=detections[selected];$('pixels').textContent=d?d.bbox[2].toFixed(1)+' px':'—';$('depth').textContent=d?.depth!=null?(d.depth*100).toFixed(1)+'% near':'—';$('scale').textContent=d?.cm_per_px!=null?d.cm_per_px.toFixed(3)+' cm/px':'—';$('coefficient').textContent=d?.coefficient!=null?'× '+d.coefficient.toFixed(3):'—';$('estimate').textContent=d?.length_m!=null?'≈ '+d.length_m.toFixed(2)+' m':'—';$('formula').textContent=d?.length_m!=null?`${d.bbox[2].toFixed(1)} px × ${d.cm_per_px.toFixed(3)} cm/px ÷ 100 = ${d.length_m.toFixed(2)} m · ${d.status.replaceAll('_',' ')}`:d?(d.status==='waiting_for_line'?`Waiting for bbox center: ${Math.abs(d.line_offset_px).toFixed(1)} px from the measurement line (tolerance ±${profile.line_tolerance_px??10} px).`:'Bottom center determines road depth. Add valid references to obtain a length.'):'Select a detected car to inspect its measurement.';$('results').replaceChildren();detections.forEach((d,i)=>{let tr=document.createElement('tr');tr.className=i===selected?'active':'';[`${i+1} · ${d.label}`,d.bbox[2].toFixed(1)+' px',d.cm_per_px?.toFixed(3)??'—',d.length_m!=null?'≈ '+d.length_m.toFixed(2)+' m':'—',(d.status??'pending').replaceAll('_',' ')].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td);});tr.onclick=()=>{selected=i;draw();inspect();};$('results').append(tr);});}
function renderReferences(){
  $('references').replaceChildren();renderRulers();updateProfileInfo();
  (profile?.references||[]).forEach((r,i)=>{
    const row=document.createElement('div');row.className='ref';
    const text=document.createElement('span');text.textContent=`Эталон ${i+1} · кадр ${r.frame}`;
    const length=document.createElement('input');length.type='number';length.min='.01';length.max='40';length.step='.01';length.value=r.length_m;length.ariaLabel=`Длина эталона ${i+1}, м`;
    length.onchange=()=>editProfile(next=>{next.references[i].length_m=Number(length.value);});
    const remove=document.createElement('button');remove.textContent='Удалить';
    remove.onclick=()=>editProfile(next=>{next.references.splice(i,1);});
    row.append(text,length,remove);$('references').append(row);
  });
}
async function editProfile(change){
  if(busy||!profile)return;
  busy=true;
  try{const next=structuredClone(profile);change(next);profile=await api('/validate-profile',next);revision++;renderReferences();}
  catch(e){status(e.message,true);renderReferences();return;}
  finally{busy=false;}
  if(media)await refresh();
}
function newProfile(size){return {version:1,image_size:size,lens:{...defaults},polygon:[],references:[],metric_rulers:[],measurement_line_x:null,line_tolerance_px:10};}
function updateProfileInfo(){
  $('profileInfo').textContent=profile?`${profileName} · ${profile.image_size.join(' × ')} px · ${profile.polygon.length} точек дороги · ${profile.references.length} эталонов · ${(profile.metric_rulers||[]).length} мерных линий`:'Калибровка не загружена. Откройте кадр или продолжите сохранённый JSON.';
  const still=!media||media.frames===1;
  for(const id of ['prev','next','play','timeline'])$(id).disabled=still;
  $('cameraFrameInfo').textContent=media?.source==='ip_camera'?`Зафиксированный кадр: ${new Date(media.captured_at).toLocaleString('ru-RU')}. Для нового снимка нажмите «Кадр с камеры».`:'Калибровка работает на неподвижном изображении в исходном разрешении.';
}
function resetView(){
  revision++;picture=null;detections=[];selected=-1;draft=[];drag=null;
  ctx.clearRect(0,0,canvas.width,canvas.height);$('empty').hidden=true;
  lensUI();setMode('select');$('lineTolerance').value=profile?.line_tolerance_px??10;inspect();renderReferences();
}
function adoptMedia(item){
  if(profile&&profile.image_size.toString()!==item.image_size.toString()){
    pendingMedia=item;
    $('resolutionMismatch').hidden=false;
    $('resolutionMessage').textContent=`Кадр ${item.image_size.join(' × ')} px не совпадает с калибровкой ${profile.image_size.join(' × ')} px. Текущие настройки сохранены. Выберите кадр нужного разрешения или начните новую калибровку на этом изображении.`;
    status('Разрешение не совпадает. Текущая калибровка не изменена.',true);return false;
  }
  if(!profile){profile=newProfile(item.image_size);profileName='Новая калибровка';}
  media=item;pendingMedia=null;$('resolutionMismatch').hidden=true;
  if(item.source==='ip_camera')$('calibrationTarget').value='ip';
  else if(item.frames>1)$('calibrationTarget').value='operator';
  resetView();$('timeline').max=item.frames-1;$('timeline').value=0;$('filename').textContent=item.name;
  return true;
}
async function sessionTask(task){
  if(busy){status('Дождитесь завершения текущей операции.');return;}
  busy=true;sessionBusy=true;playing=false;clearTimeout(timer);
  let redraw=false;
  try{redraw=await task();}
  catch(e){status(e.message,true);}
  finally{busy=false;sessionBusy=false;}
  if(redraw&&media)await refresh();
}
$('media').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  await sessionTask(async()=>{
    if(profile&&!await finishPendingRoad())return false;
    status('Открываем изображение или видео…');const fd=new FormData();fd.append('file',file);
    return adoptMedia(await api('/media',fd));
  });e.target.value='';
};
$('cameraFrame').onclick=()=>sessionTask(async()=>{
  if(profile&&!await finishPendingRoad())return false;
  status('Получаем кадр боковой камеры…');
  return adoptMedia(await api('/camera-frame',{}));
});
$('useNewImage').onclick=()=>sessionTask(async()=>{
  if(!pendingMedia)return false;
  profile=newProfile(pendingMedia.image_size);profileName='Новая калибровка';
  return adoptMedia(pendingMedia);
});
$('keepCalibration').onclick=()=>{pendingMedia=null;$('resolutionMismatch').hidden=true;};
$('newCalibration').onclick=()=>sessionTask(async()=>{
  media=null;profile=null;pendingMedia=null;profileName='Новая калибровка';
  $('resolutionMismatch').hidden=true;resetView();$('empty').hidden=false;$('filename').textContent='NO MEDIA LOADED';
  status('Откройте кадр с камеры или файл. Сохранённые настройки станции не изменены.');return false;
});
async function loadProfile(data,name){
  const next=await api('/validate-profile',data.profile||data);
  if(media&&next.image_size.toString()!==media.image_size.toString())throw Error('Размер JSON не совпадает с изображением. Настройки сохранены. Нажмите «Новая сессия», затем загрузите JSON и кадр нужного разрешения.');
  profile=next;profileName=name;pendingMedia=null;$('resolutionMismatch').hidden=true;resetView();
  if(!media){$('empty').hidden=false;status('Калибровка загружена. Откройте кадр с камеры или изображение того же разрешения, чтобы продолжить.');}
  return !!media;
}
async function stationProfileRequest(method,body,source=$('calibrationTarget').value){
  const target=source==='ip'?'/api/ip/calibration':'/api/workbench/operator-calibration';
  const response=await fetch(target,{method,cache:'no-store',...(body?{headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});
  const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
$('loadSaved').onclick=()=>sessionTask(async()=>{
  const target=$('calibrationTarget').value;
  return loadProfile(await stationProfileRequest('GET',null,target),target==='ip'?'Сохранённая калибровка IP-камер':'Сохранённая калибровка видеозаписей');
});
$('applyCalibration').onclick=()=>sessionTask(async()=>{
  const target=$('calibrationTarget').value;
  if(!media||!profile)throw Error('Сначала откройте изображение и проверьте калибровку.');
  if(!await finishPendingRoad())return false;
  if(profile.polygon.length<4||!(profile.references.length||profile.metric_rulers?.length))throw Error('Нужна дорога и хотя бы один эталон или мерная линия.');
  const next=await api('/validate-profile',profile);
  await stationProfileRequest('POST',next,target);
  profile=next;renderReferences();
  if(typeof window!=='undefined')window.parent.postMessage({type:'station-calibration-saved',target,profile:next},location.origin);
  status('Калибровка сохранена для выбранного источника. Для IP-камер подключите камеры с новой настройкой.');return false;
});
$('grid').onchange=draw;$('resetLens').onclick=()=>{if(!ready())return;profile.lens={...defaults};lensUI();invalidateLens();};
$('drawRoad').onclick=()=>{if(!ready())return;playing=false;draft=[];setMode('road');status('Click boundary points in order. Finish with at least 4 dots.');};
$('undoRoad').onclick=()=>{if(mode==='road'){draft.pop();draw();}};
$('finishRoad').onclick=async()=>{if(ready()&&await finishPendingRoad())await refresh();};
$('drawBox').onclick=async()=>{if(ready()&&await finishPendingRoad()){playing=false;setMode('box');status('Drag a box tightly around the full car. Its bottom center must lie on the road.');}};
$('detect').onclick=async()=>{if(!ready()||!await finishPendingRoad())return;playing=false;setMode('select');refresh(true);};
$('confidence').oninput=()=>$('confValue').textContent=Number($('confidence').value).toFixed(2);
function lengthLabel(){$('lengthValue').textContent=Number($('length').value).toFixed(2)+' m';}$('length').oninput=lengthLabel;
for(const [id,delta] of [['lengthDown',-.01],['lengthUp',.01]])$(id).onclick=()=>{$('length').value=Number($('length').value)+delta;lengthLabel();};
$('addRef').onclick=async()=>{if(!ready()||!await finishPendingRoad())return;if(selected<0){status('Select a detected car or draw its bbox first.',true);return;}if(profile.measurement_line_x!=null&&Math.abs(detections[selected].bbox[0]+detections[selected].bbox[2]/2-profile.measurement_line_x)>(profile.line_tolerance_px??10)){status('Move to a frame where the bbox center meets the measurement line before adding a reference.',true);return;}let r={bbox:detections[selected].bbox,length_m:Number($('length').value),frame:Number($('timeline').value)};let next=structuredClone(profile);next.references.push(r);try{await api('/validate-profile',next);profile=next;await refresh();}catch(e){status(e.message,true);}};
async function seek(n,detect=false){if(!ready())return;detections=[];selected=-1;revision++;$('timeline').value=Math.max(0,Math.min(media.frames-1,n));await refresh(detect);}
$('timeline').onchange=()=>{playing=false;seek(Number($('timeline').value));};$('prev').onclick=()=>{playing=false;seek(Number($('timeline').value)-1);};$('next').onclick=()=>{playing=false;seek(Number($('timeline').value)+1);};
$('play').onclick=async()=>{if(playing){playing=false;return;}if(!ready()||media.frames<2||!await finishPendingRoad())return;
if(typeof window!=='undefined'&&window.parent!==window){window.parent.postMessage({type:'station-play',media,profile,frame:Number($('timeline').value)},location.origin);return;}
playing=true;$('play').textContent='Pause';const start=Date.now(), first=Number($('timeline').value);while(playing){let n=Math.max(Number($('timeline').value)+1,first+Math.floor((Date.now()-start)*(media.fps||25)/1000));if(n>=media.frames){playing=false;break;}await seek(n,true);}$('play').textContent='Play';};
function download(name,data){let url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save').onclick=()=>sessionTask(async()=>{
  if(!profile)throw Error('Сначала загрузите JSON или откройте изображение.');
  if(!await finishPendingRoad())return false;
  const validated=await api('/validate-profile',profile);
  download('roadscale-calibration.json',validated);status('JSON сохранён. Его можно загрузить и продолжить редактирование.');return false;
});
$('import').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  await sessionTask(async()=>loadProfile(JSON.parse(await file.text()),file.name));e.target.value='';
};
$('exportResults').onclick=()=>{if(!ready())return;download('roadscale-measurements.json',{media:media.name,frame:Number($('timeline').value),model:'yolo26n.pt',method:'empirical_bbox_approximation',profile,detections});};


// A draft is never silently discarded when leaving road drawing.
async function finishPendingRoad() {
  if (mode === 'ruler') return finishRuler();
  if (mode !== 'road') return true;
  if (!draft.length && profile.polygon.length >= 4) {
    setMode('select');
    return true;
  }
  if (draft.length < 4) {
    status('Place at least four road dots before continuing. Your points are still on the image.', true);
    return false;
  }
  const polygon = structuredClone(draft);
  const changed = JSON.stringify(polygon) !== JSON.stringify(profile.polygon);
  const next = {...profile, polygon, references: changed ? [] : profile.references, metric_rulers: changed ? [] : (profile.metric_rulers||[])};
  try {
    await api('/validate-profile', next);
    profile = next;
    draft = [];
    setMode('select');
    renderReferences();
    return true;
  } catch (e) {
    status(e.message, true);
    return false;
  }
}

function drawDraftAndLine(s) {
  ctx.save();
  if (mode === 'road' && draft.length) {
    ctx.beginPath();
    draft.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    if (draft.length >= 4) ctx.closePath();
    ctx.strokeStyle = '#80e6bd';
    ctx.fillStyle = '#65e6aa20';
    ctx.lineWidth = 2*s;
    ctx.fill(); ctx.stroke();
    draft.forEach(([x,y],i) => {
      ctx.beginPath(); ctx.arc(x,y,5*s,0,Math.PI*2);
      ctx.fillStyle = '#80e6bd'; ctx.fill();
      ctx.fillText(i+1,x+8*s,y-8*s);
    });
  }
  if (profile.measurement_line_x != null) {
    const x = profile.measurement_line_x, tolerance = profile.line_tolerance_px ?? 10;
    ctx.fillStyle = '#d0a0ff25';
    ctx.fillRect(x-tolerance,0,tolerance*2,canvas.height);
    ctx.strokeStyle = '#d0a0ff'; ctx.lineWidth = 2*s;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke();
    ctx.fillStyle = '#d0a0ff';
    ctx.fillText('CENTER MEASUREMENT LINE',Math.min(x+8*s,canvas.width-220*s),30*s);
  }
  ctx.restore();
}

function updateLineLabel() {
  $('lineInfo').textContent = profile?.measurement_line_x != null
    ? `Line x = ${profile.measurement_line_x.toFixed(1)} px · drag the purple line`
    : 'Line disabled: measurements are available across the road.';
  $('lineToleranceValue').textContent = `± ${profile?.line_tolerance_px ?? 10} px`;
}
$('placeLine').onclick = async () => {
  if (!ready() || !await finishPendingRoad()) return;
  playing = false;
  setMode('line');
  status('Click where cars appear side-on. Measurement uses the center of the bbox, not its front edge.');
};
$('removeLine').onclick = async () => {
  if (!ready() || !await finishPendingRoad()) return;
  profile.measurement_line_x = null;
  setMode('select');
  refresh();
};
$('lineTolerance').oninput = () => {
  if (!profile) return;
  if(sessionBusy){$('lineTolerance').value=profile.line_tolerance_px??10;return;}
  playing = false;
  profile.line_tolerance_px = Number($('lineTolerance').value);
  updateLineLabel();
  draw();
};
$('lineTolerance').onchange = () => { if (media) refresh(); };
updateLineLabel();

let stationSending = false;
async function sendToStation() {
  if (stationSending || !ready() || !await finishPendingRoad()) return;
  const d = detections[selected];
  if (!d) { status('Select a car before sending it to the operator.', true); return; }
  stationSending = true;
  $('sendStation').disabled = true;
  try {
    const response = await fetch('/api/station/capture', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({media_id:media.id,profile,frame:Number($('timeline').value),
        bbox:d.bbox,label:d.label,source:d.source??'manual',actor:localStorage.getItem('ferryOperator')||'Оператор'})
    });
    const record = await response.json();
    if (!response.ok) throw Error(typeof record.detail === 'string' ? record.detail : JSON.stringify(record.detail));
    status('Автомобиль сохранён в очереди оператора. Проверьте категорию, номер и тариф.');
    window.parent.postMessage({type:'station-capture',id:record.id},location.origin);
    return record;
  } catch (e) { status(e.message, true); }
  finally { stationSending = false; $('sendStation').disabled = false; }
}
$('sendStation').onclick = sendToStation;


// All marks use original corrected-image coordinates, regardless of CSS display size.
function drawRulers(s){
  const rulers=[...(profile.metric_rulers||[])];
  if(mode==='ruler'&&draft.length)rulers.push({points:draft,step_m:Number($('rulerStep').value)});
  for(const ruler of rulers){
    ctx.strokeStyle='#ffb580';ctx.fillStyle='#ffb580';ctx.lineWidth=2*s;ctx.beginPath();
    ruler.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
    ruler.points.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,4*s,0,Math.PI*2);ctx.fill();ctx.fillText((i*ruler.step_m).toFixed(2)+' m',x+6*s,y-7*s);});
  }
}
function renderRulers(){
  $('rulers').replaceChildren();
  (profile?.metric_rulers||[]).forEach((r,i)=>{
    const row=document.createElement('div');row.className='ref';
    const text=document.createElement('span');text.textContent=`Ruler ${i+1}: ${r.points.length-1} × ${r.step_m} m`;
    const remove=document.createElement('button');remove.textContent='Remove';
    remove.onclick=()=>editProfile(next=>{next.metric_rulers.splice(i,1);});
    const spacing=document.createElement('input');spacing.type='number';spacing.min='.01';spacing.max='20';spacing.step='.01';spacing.value=r.step_m;spacing.ariaLabel=`Шаг мерной линии ${i+1}, м`;
    spacing.onchange=()=>editProfile(next=>{next.metric_rulers[i].step_m=Number(spacing.value);});
    row.append(text,spacing,remove);$('rulers').append(row);
  });
}
async function finishRuler(){
  if(mode!=='ruler')return true;
  if(draft.length<3){status('Click at least three consecutive measured marks. The draft is still on the image.',true);return false;}
  const step=Number($('rulerStep').value);
  if(!Number.isFinite(step)||step<=0){status('Enter the actual positive distance between marks.',true);return false;}
  const next=structuredClone(profile);next.metric_rulers??=[];
  next.metric_rulers.push({points:structuredClone(draft),step_m:step});
  try{await api('/validate-profile',next);profile=next;draft=[];setMode('select');renderReferences();return true;}
  catch(e){status(e.message,true);return false;}
}
$('drawRuler').onclick=async()=>{
  if(!ready()||!await finishPendingRoad())return;
  if(profile.polygon.length<4){status('Draw the road first.',true);return;}
  draft=[];setMode('ruler');status('Click each real measured mark in order along the lane. Do not estimate metre spacing by eye.');
};
$('finishRuler').onclick=async()=>{if(ready()&&await finishRuler())await refresh();};
$('undoRuler').onclick=()=>{if(mode==='ruler'){draft.pop();draw();}};
$('cancelRuler').onclick=()=>{if(mode==='ruler'){draft=[];setMode('select');}};

updateProfileInfo();
