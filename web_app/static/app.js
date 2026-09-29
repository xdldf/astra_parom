const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
let media=null, profile=null, picture=null, detections=[], selected=-1, mode='select', draft=[], drag=null, busy=false, playing=false, revision=0, timer;
let profileName='Новая калибровка', pendingMedia=null, sessionBusy=false;
let savedProfiles={}, sourceDrafts={}, undoHistory=[], undoGroup=null;
let currentTarget='ip';
let detectorPreferences={detector_model:'yolo26m',detector_imgsz:640};
const defaults={k1:0,k2:0,focal:1,zoom:1,cx:.5,cy:.5,tilt_deg:0};
const controls=[['k1','Радиальная коррекция',-.8,.8,.005],['k2','Коррекция краёв',-.5,.5,.005],['tilt_deg','Наклон, градусы',-30,30,.1],['focal','Фокусное расстояние',.4,2,.01],['zoom','Масштаб коррекции',.5,2,.01],['cx','Центр объектива по горизонтали',.2,.8,.005],['cy','Центр объектива по вертикали',.2,.8,.005]];
function status(text,error=false){updateSaveState();$('status').textContent=text;$('status').className=error?'error':'';}
async function api(path,body){let r=await fetch('/api/workbench'+path,{method:'POST',headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body instanceof FormData?body:JSON.stringify(body)});let data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
function ready(){if(!media){status('Откройте кадр с камеры, изображение или видео. Загруженная калибровка сохранится.',true);return false;}if(busy){status('Дождитесь обработки текущего кадра.');return false;}return true;}
function setMode(value){mode=value;$('mode').textContent={road:'Отмечайте границу дороги по порядку',box:'Обведите автомобиль целиком',select:'Выберите автомобиль · точки дороги и линию можно перетаскивать',line:'Нажмите на кадр, чтобы поставить линию измерения',ruler:'Отметьте мерные точки по порядку, затем нажмите «Готово»'}[value];draw();}
function lensUI(){for(const [id] of controls){$(id).value=profile?.lens[id]??defaults[id];$(id+'Value').textContent=Number($(id).value).toFixed(3);}}
for(const [id,label,min,max,step] of controls){let l=document.createElement('label');l.className='sliderlabel';l.innerHTML=`${label}<output id="${id}Value"></output><input type="range" id="${id}" min="${min}" max="${max}" step="${step}">`;$('lensControls').append(l);$(id).oninput=()=>{if(sessionBusy){lensUI();return;}if(!media)return;checkpoint('lens');profile.lens[id]=Number($(id).value);$(id+'Value').textContent=Number($(id).value).toFixed(3);invalidateLens();};}
lensUI();
function invalidateLens(){playing=false;profile.polygon=[];profile.references=[];profile.metric_rulers=[];profile.survey_calibration=null;profile.measurement_line_x=null;draft=[];detections=[];selected=-1;revision++;setMode('select');renderReferences();inspect();clearTimeout(timer);timer=setTimeout(()=>refresh(false),180);}
async function refresh(detect=false){if(!media)return;if(busy){clearTimeout(timer);timer=setTimeout(()=>refresh(detect),200);return;}busy=true;detectorUI();const rev=revision;const frame=Number($('timeline').value);$('detect').disabled=true;try{status(detect?`Запускаем ${profile.detector_model??'yolo26n'} на исправленном кадре… При первом запуске загружаются веса.`:'Обновляем исправленный кадр…');const data=await api('/frame/'+media.id,{profile,frame,detect,confidence:Number($('confidence').value),boxes:detect?[]:detections.map(d=>d.bbox)});const img=new Image();await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=reject;img.src='data:image/jpeg;base64,'+data.image;});if(rev!==revision)return;picture=img;const previous=detections;detections=data.detections.map((d,i)=>detect?{...d,source:profile.detector_model??'yolo26n'}:{...d,label:previous[i]?.label??d.label,confidence:previous[i]?.confidence??null,source:previous[i]?.source??'manual'});selected=Math.min(selected,detections.length-1);if(selected<0&&detections.length)selected=0;canvas.width=img.width;canvas.height=img.height;applyViewZoom();$('empty').hidden=true;$('frameLabel').textContent=media.frames===1?`Снимок · ${profile.image_size.join(' × ')} px`:`Кадр ${frame} / ${media.frames-1}`;draw();inspect();renderReferences();status(data.scale?.status==='ruler_calibrated'?'Мерные линии активны. Результаты вне покрытия зависят от выбранного режима измерения.':data.scale?.status==='verified_local'?'Проверенные автомобили: местный масштаб только в полосе эталонов.':data.scale?.status==='depth_calibrated'?'Масштаб учитывает положение на дороге. Выберите автомобиль для проверки.':data.scale?'Один эталон: постоянный масштаб. Добавьте автомобиль на другой глубине дороги.':detect?`Найдено автомобилей: ${detections.length}. Выберите автомобиль для проверки.`:'Отметьте дорогу, затем добавьте мерные линии или автомобили известной длины.');showCalibrationDiagnostics(data.scale);}catch(e){status(e.message,true);playing=false;}finally{busy=false;$('detect').disabled=false;detectorUI();}}

function showCalibrationDiagnostics(scale){
  const d=scale?.diagnostics;
  if(!d)return;
  if(d.survey){status(`Метровые отметки: контроль ${(100*d.ruler_check_error_m).toFixed(1)} см; перенос на дорогу: ${(100*d.ground_check_error_m).toFixed(1)} см. Допуск ±${100*d.target_tolerance_m} см. Точность автомобиля не подтверждена.`,Math.max(d.ruler_check_error_m,d.ground_check_error_m)>d.target_tolerance_m);return;}
  const error=(100*d.reference_max_abs_error_m).toFixed(1), tolerance=d.target_tolerance_m??.1;
  const duplicates=d.duplicates_ignored?` Повторных эталонов исключено: ${d.duplicates_ignored}.`:'';
  status(`Проверка эталонов: максимальное расхождение ${error} см; в пределах ±${100*tolerance} см: ${d.references_within_target}/${d.unique_reference_count}.${duplicates} Точность на независимых автомобилях не подтверждена.`,d.reference_max_abs_error_m>tolerance);
}
function point(e){let r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height-1,(e.clientY-r.top)*canvas.height/r.height))];}
function draw(){if(!picture)return;ctx.drawImage(picture,0,0);const s=canvas.width/1000;ctx.lineWidth=2*s;ctx.font=`${13*s}px system-ui`;if($('grid').checked){ctx.strokeStyle='#ffffff35';ctx.lineWidth=s;for(let n=1;n<10;n++){ctx.beginPath();ctx.moveTo(0,n*canvas.height/10);ctx.lineTo(canvas.width,n*canvas.height/10);ctx.stroke();ctx.beginPath();ctx.moveTo(n*canvas.width/10,0);ctx.lineTo(n*canvas.width/10,canvas.height);ctx.stroke();}}let p=profile.polygon;if(p.length){ctx.beginPath();p.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle='#65e6aa20';ctx.fill();ctx.strokeStyle='#80e6bd';ctx.lineWidth=2*s;ctx.stroke();p.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,5*s,0,Math.PI*2);ctx.fillStyle='#80e6bd';ctx.fill();ctx.fillText(i+1,x+8*s,y-8*s);});}drawDraftAndLine(s);drawRulers(s);detections.forEach((d,i)=>{let [x,y,w,h]=d.bbox;ctx.strokeStyle=i===selected?'#ffe084':'#78c8ff';ctx.lineWidth=(i===selected?3:1.5)*s;ctx.strokeRect(x,y,w,h);ctx.beginPath();ctx.moveTo(x+w/2,y);ctx.lineTo(x+w/2,y+h);ctx.stroke();ctx.beginPath();ctx.arc(x+w/2,y+h/2,4*s,0,Math.PI*2);ctx.fillStyle=d.at_measurement_line?'#80e6bd':ctx.strokeStyle;ctx.fill();ctx.beginPath();ctx.moveTo(x,y+h);ctx.lineTo(x+w,y+h);ctx.stroke();ctx.beginPath();ctx.arc(...(d.road_contact||[x+w/2,y+h]),5*s,0,Math.PI*2);ctx.fillStyle=ctx.strokeStyle;ctx.fill();let text=`${i+1} · ${d.label} ${d.confidence!=null?Math.round(d.confidence*100)+'% ':''}${d.length_m!=null?'≈ '+d.length_m.toFixed(2)+' м':''}`;let tw=ctx.measureText(text).width;ctx.fillStyle='#09151ee8';ctx.fillRect(x,Math.max(0,y-24*s),tw+12*s,24*s);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(text,x+6*s,Math.max(16*s,y-7*s));if(i===selected&&d.depth!=null){ctx.save();ctx.setLineDash([6*s,5*s]);ctx.beginPath();const section=d.road_cross_section;if(section){ctx.moveTo(...section.far);ctx.lineTo(...section.near);}else{ctx.moveTo(x,y+h);ctx.lineTo(x+w,y+h);}ctx.strokeStyle='#ffe08480';ctx.stroke();if(d.projected_ground_span_px){ctx.beginPath();ctx.moveTo(...d.projected_ground_span_px[0]);ctx.lineTo(...d.projected_ground_span_px[1]);ctx.strokeStyle='#80e6bd';ctx.stroke();}ctx.restore();}});if(drag?.type==='box'){let [x,y]=drag.start,[ex,ey]=drag.end;ctx.strokeStyle='#ffe084';ctx.strokeRect(x,y,ex-x,ey-y);}}
canvas.onpointerdown=e=>{if(!ready()||!picture)return;playing=false;let p=point(e);canvas.setPointerCapture(e.pointerId);if(mode==='road'||mode==='ruler'){draft.push(p);draw();return;}if(mode==='line'){checkpoint();profile.measurement_line_x=p[0];setMode('select');refresh();return;}if(mode==='box'){drag={type:'box',start:p,end:p};return;}let near=profile.polygon.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<12*canvas.width/canvas.clientWidth);if(near>=0){checkpoint();drag={type:'vertex',index:near,old:structuredClone(profile)};return;}for(let r=0;r<(profile.metric_rulers||[]).length;r++){const index=profile.metric_rulers[r].points.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<12*canvas.width/canvas.clientWidth);if(index>=0){checkpoint();drag={type:'ruler-point',ruler:r,index,old:structuredClone(profile)};return;}}if(profile.measurement_line_x!=null&&Math.abs(p[0]-profile.measurement_line_x)<10*canvas.width/canvas.clientWidth){checkpoint();drag={type:'line',old:structuredClone(profile)};return;}selected=detections.findIndex(d=>p[0]>=d.bbox[0]&&p[0]<=d.bbox[0]+d.bbox[2]&&p[1]>=d.bbox[1]&&p[1]<=d.bbox[1]+d.bbox[3]);draw();inspect();};
canvas.onpointermove=e=>{if(!drag)return;let p=point(e);if(drag.type==='box')drag.end=p;else if(drag.type==='line')profile.measurement_line_x=p[0];else if(drag.type==='ruler-point')profile.metric_rulers[drag.ruler].points[drag.index]=p;else profile.polygon[drag.index]=p;draw();};
canvas.onpointerup=async()=>{if(!drag)return;let d=drag;drag=null;if(d.type==='box'){let [x,y]=d.start,[ex,ey]=d.end;let b=[Math.min(x,ex),Math.min(y,ey),Math.abs(ex-x),Math.abs(ey-y)];if(b[2]>3&&b[3]>3){detections.push({bbox:b,label:'manual car',source:'manual'});selected=detections.length-1;setMode('select');await refresh();}}else{try{await api('/validate-profile',profile);await refresh();}catch(e){profile=d.old;status(e.message,true);draw();}}};
canvas.onpointercancel=()=>{if(drag?.old)profile=drag.old;drag=null;draw();};
function inspect(){updateLineLabel();const d=detections[selected];$('pixels').textContent=d?d.bbox[2].toFixed(1)+' px':'—';$('depth').textContent=d?.depth!=null?(d.depth*100).toFixed(1)+'% дороги':'—';$('scale').textContent=d?.cm_per_px!=null?d.cm_per_px.toFixed(3)+' см/px':'—';$('coefficient').textContent=d?.coefficient!=null?'× '+d.coefficient.toFixed(3):'—';$('estimate').textContent=d?.length_m!=null?'≈ '+d.length_m.toFixed(2)+' м':'—';$('formula').textContent=d?.length_m!=null?`${d.bbox[2].toFixed(1)} px × ${d.cm_per_px.toFixed(3)} cm/px ÷ 100 = ${d.length_m.toFixed(2)} m · ${measurementStatus(d.status)}`:d?(d.status==='calibration_review'?'Проверка геометрии превышает допуск; длина не назначена.':d.status==='waiting_for_line'?`До линии измерения: ${Math.abs(d.line_offset_px).toFixed(1)} px (допуск ±${profile.line_tolerance_px??10} px).`:'Положение определяется по низу рамки или её пересечению с дорогой. Для длины нужна действующая калибровка.'):'Выберите рамку автомобиля на кадре.';$('results').replaceChildren();detections.forEach((d,i)=>{let tr=document.createElement('tr');tr.className=i===selected?'active':'';[`${i+1} · ${d.label}`,d.bbox[2].toFixed(1)+' px',d.cm_per_px?.toFixed(3)??'—',d.length_m!=null?'≈ '+d.length_m.toFixed(2)+' м':'—',measurementStatus(d.status)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td);});tr.onclick=()=>{selected=i;draw();inspect();};$('results').append(tr);});}
function renderReferences(){
  $('references').replaceChildren();renderRulers();updateProfileInfo();
  (profile?.references||[]).forEach((r,i)=>{
    const row=document.createElement('div');row.className='ref';
    const text=document.createElement('span');text.textContent=`Эталон ${i+1} · кадр ${r.frame}`;
    const length=document.createElement('input');length.type='number';length.min='.01';length.max='40';length.step='.01';length.value=r.length_m;length.ariaLabel=`Длина эталона ${i+1}, м`;
    length.disabled=!!r.vehicle_id;
    if(r.vehicle_id)text.textContent=`Проверенный автомобиль · ${r.verified_by} · ${r.verified_at}`;
    length.onchange=()=>editProfile(next=>{next.references[i].length_m=Number(length.value);});
    const remove=document.createElement('button');remove.textContent='Удалить';
    remove.disabled=!!r.vehicle_id;
    if(r.vehicle_id)remove.textContent='Изменяется в списке автомобилей';
    remove.onclick=()=>editProfile(next=>{next.references.splice(i,1);});
    row.append(text,length,remove);$('references').append(row);
  });
}
async function editProfile(change){
  if(busy||!profile)return;
  checkpoint();busy=true;detectorUI();
  try{const next=structuredClone(profile);change(next);profile=await api('/validate-profile',next);revision++;renderReferences();}
  catch(e){status(e.message,true);renderReferences();return;}
  finally{busy=false;detectorUI();}
  if(media)await refresh();
}
function newProfile(size){return {version:1,image_size:size,lens:{...defaults},polygon:[],references:[],metric_rulers:[],measurement_line_x:null,line_tolerance_px:10,accuracy_tolerance_m:.1,measurement_mode:'estimate',...detectorPreferences};}
function updateProfileInfo(){
  updateSaveState();
  $('profileInfo').textContent=profile?`${profileName} · ${profile.image_size.join(' × ')} px · ${profile.polygon.length} точек дороги · ${profile.references.length} эталонов · ${(profile.metric_rulers||[]).length} мерных линий`:'Калибровка не загружена. Откройте кадр или продолжите сохранённый JSON.';
  const still=!media||media.frames===1;
  for(const id of ['prev','next','play','timeline'])$(id).disabled=still;
  $('cameraFrameInfo').textContent=media?.source==='ip_camera'?`Зафиксированный кадр: ${new Date(media.captured_at).toLocaleString('ru-RU')}. Для нового снимка нажмите «Кадр с камеры».`:'Калибровка работает на неподвижном изображении в исходном разрешении.';
}
function resetView(){
  revision++;picture=null;detections=[];selected=-1;draft=[];drag=null;
  ctx.clearRect(0,0,canvas.width,canvas.height);$('empty').hidden=true;
  lensUI();detectorUI();measurementUI();setMode('select');$('lineTolerance').value=profile?.line_tolerance_px??10;inspect();renderReferences();
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
  // Opening media does not silently change the destination of the settings.
  if(!$('calibrationTarget').value||$('calibrationTarget').value==='0')$('calibrationTarget').value='ip';
  resetView();$('timeline').max=item.frames-1;$('timeline').value=0;$('filename').textContent=item.name;
  return true;
}
async function sessionTask(task){
  if(busy){status('Дождитесь завершения текущей операции.');return;}
  busy=true;sessionBusy=true;playing=false;clearTimeout(timer);detectorUI();
  let redraw=false;
  try{redraw=await task();}
  catch(e){status(e.message,true);}
  finally{busy=false;sessionBusy=false;detectorUI();updateSaveState();}
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
  checkpoint();media=null;profile=null;pendingMedia=null;profileName='Новая калибровка';
  $('resolutionMismatch').hidden=true;resetView();$('empty').hidden=false;$('filename').textContent='Кадр не открыт';
  status('Откройте кадр с камеры или файл. Сохранённые настройки станции не изменены.');return false;
});
async function loadProfile(data,name){
  const next=await api('/approved-references',await api('/validate-profile',data.profile||data));
  if(media&&next.image_size.toString()!==media.image_size.toString())throw Error('Размер JSON не совпадает с изображением. Настройки сохранены. Нажмите «Новая сессия», затем загрузите JSON и кадр нужного разрешения.');
  checkpoint();profile=next;profileName=name;pendingMedia=null;$('resolutionMismatch').hidden=true;resetView();
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
async function loadSavedProfile(target){
  const data=await stationProfileRequest('GET',null,target);
  const redraw=await loadProfile(data,target==='ip'?'Сохранённая калибровка IP-камер':'Сохранённая калибровка видеозаписей');
  savedProfiles[target]=structuredClone(profile);currentTarget=target;
  updateSaveState();return redraw;
}
$('loadSaved').onclick=()=>sessionTask(()=>loadSavedProfile($('calibrationTarget').value));
$('applyCalibration').onclick=()=>sessionTask(async()=>{
  const target=$('calibrationTarget').value;
  if(!profile)throw Error('Загрузите сохранённую калибровку или JSON. Для новой настройки откройте кадр.');
  if(!await finishPendingRoad())return false;
  if(profile.polygon.length<4||!(profile.references.length||profile.metric_rulers?.length||profile.survey_calibration))throw Error('Нужна дорога и хотя бы один эталон или мерная линия.');
  const next=await api('/approved-references',await api('/validate-profile',profile));
  status('Применяем настройку. Если камеры работают, проверяем модель и переподключаем их…');
  const result=await stationProfileRequest('POST',next,target);
  // Use the server's canonical profile, including merged verified references.
  profile=result.profile||result;
  if(!profile.image_size)profile=next;
  savedProfiles[target]=structuredClone(profile);profileName='Сохранённая калибровка '+(target==='ip'?'IP-камер':'видеозаписей');
  renderReferences();
  if(typeof window!=='undefined')window.parent.postMessage({type:'station-calibration-saved',target,profile},location.origin);
  status(result.restarted?'Калибровка сохранена и применена. Камеры переподключены с новой настройкой.':target==='ip'?'Калибровка сохранена. При подключении камеры будут использовать эту модель.':'Калибровка сохранена для видеозаписей.');return false;
});
$('grid').onchange=draw;$('resetLens').onclick=()=>{if(!ready())return;checkpoint();profile.lens={...defaults};lensUI();invalidateLens();};
$('drawRoad').onclick=()=>{if(!ready())return;playing=false;draft=[];setMode('road');status('Отмечайте границу по порядку. Для завершения нужны минимум 4 точки.');};
$('undoRoad').onclick=()=>{if(mode==='road'){draft.pop();draw();}};
$('finishRoad').onclick=async()=>{if(ready()&&await finishPendingRoad())await refresh();};
$('drawBox').onclick=async()=>{if(ready()&&await finishPendingRoad()){playing=false;setMode('box');status('Обведите автомобиль целиком. Для оценки достаточно пересечения рамки с дорогой; сначала используется низ рамки.');}};
$('detect').onclick=async()=>{if(!ready()||!await finishPendingRoad())return;playing=false;setMode('select');refresh(true);};
$('confidence').oninput=()=>$('confValue').textContent=Number($('confidence').value).toFixed(2);
function lengthLabel(){$('lengthValue').textContent=Number($('length').value).toFixed(2)+' м';$('lengthNumber').value=Number($('length').value).toFixed(2);}$('length').oninput=lengthLabel;
$('lengthNumber').onchange=()=>{const value=Number($('lengthNumber').value);if(!Number.isFinite(value)||value<1||value>25){status('Длина эталона должна быть от 1 до 25 м.',true);return;}$('length').value=value;lengthLabel();};
for(const [id,delta] of [['lengthDown',-.01],['lengthUp',.01]])$(id).onclick=()=>{$('length').value=Number($('length').value)+delta;lengthLabel();};
$('addRef').onclick=async()=>{if(!ready()||!await finishPendingRoad())return;if(selected<0){status('Выберите обнаруженный автомобиль или нарисуйте рамку.',true);return;}if(profile.measurement_line_x!=null&&Math.abs(detections[selected].bbox[0]+detections[selected].bbox[2]/2-profile.measurement_line_x)>(profile.line_tolerance_px??10)){status('Перед добавлением эталона выберите кадр, где центр рамки попадает на линию измерения.',true);return;}let r={bbox:detections[selected].bbox,length_m:Number($('length').value),frame:Number($('timeline').value)};let next=structuredClone(profile);const existing=next.references.findIndex(q=>!q.vehicle_id&&q.frame===r.frame&&q.bbox.every((v,i)=>v===r.bbox[i]));if(existing>=0)next.references[existing]=r;else next.references.push(r);try{await api('/validate-profile',next);checkpoint();profile=next;await refresh();}catch(e){status(e.message,true);}};
async function seek(n,detect=false){if(!ready())return;detections=[];selected=-1;revision++;$('timeline').value=Math.max(0,Math.min(media.frames-1,n));await refresh(detect);}
$('timeline').onchange=()=>{playing=false;seek(Number($('timeline').value));};$('prev').onclick=()=>{playing=false;seek(Number($('timeline').value)-1);};$('next').onclick=()=>{playing=false;seek(Number($('timeline').value)+1);};
$('play').onclick=async()=>{if(playing){playing=false;return;}if(!ready()||media.frames<2||!await finishPendingRoad())return;
if(typeof window!=='undefined'&&window.parent!==window){window.parent.postMessage({type:'station-play',media,profile,frame:Number($('timeline').value)},location.origin);return;}
playing=true;$('play').textContent='Ⅱ Пауза';const start=Date.now(), first=Number($('timeline').value);while(playing){let n=Math.max(Number($('timeline').value)+1,first+Math.floor((Date.now()-start)*(media.fps||25)/1000));if(n>=media.frames){playing=false;break;}await seek(n,true);}$('play').textContent='▶ Пуск';};
function download(name,data){let url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save').onclick=()=>sessionTask(async()=>{
  if(!profile)throw Error('Сначала загрузите JSON или откройте изображение.');
  if(!await finishPendingRoad())return false;
  const validated=await api('/approved-references',await api('/validate-profile',profile));
  profile=validated;renderReferences();
  download('roadscale-calibration.json',validated);status('JSON сохранён. Его можно загрузить и продолжить редактирование.');return false;
});
$('import').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  await sessionTask(async()=>loadProfile(JSON.parse(await file.text()),file.name));e.target.value='';
};
$('exportResults').onclick=()=>sessionTask(async()=>{
  if(!media||!profile||!await finishPendingRoad())return false;
  const next=await api('/approved-references',profile),frame=Number($('timeline').value);
  const result=await api('/frame/'+media.id,{profile:next,frame,detect:false,boxes:detections.map(d=>d.bbox)});
  download('roadscale-measurements.json',{media:media.name,frame,model:(profile.detector_model??'yolo26n')+'.pt',imgsz:profile.detector_imgsz??640,method:'empirical_bbox_approximation',profile:next,detections:result.detections});return false;
});


// A draft is never silently discarded when leaving road drawing.
async function finishPendingRoad() {
  if (mode === 'ruler') return finishRuler();
  if (mode !== 'road') return true;
  if (!draft.length && profile.polygon.length >= 4) {
    setMode('select');
    return true;
  }
  if (draft.length < 4) {
    status('Добавьте минимум четыре точки дороги. Ваши точки сохранены на кадре.', true);
    return false;
  }
  const polygon = structuredClone(draft);
  const changed = JSON.stringify(polygon) !== JSON.stringify(profile.polygon);
  const next = {...profile, polygon, references: changed ? [] : profile.references, metric_rulers: changed ? [] : (profile.metric_rulers||[])};
  try {
    await api('/validate-profile', next);
    checkpoint();profile = next;
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
    ctx.fillText('ЛИНИЯ ИЗМЕРЕНИЯ',Math.min(x+8*s,canvas.width-220*s),30*s);
  }
  ctx.restore();
}

function updateLineLabel() {
  $('lineInfo').textContent = profile?.measurement_line_x != null
    ? `Линия x = ${profile.measurement_line_x.toFixed(1)} px · перетащите фиолетовую линию`
    : 'Измерение доступно по всей дороге.';
  $('lineToleranceValue').textContent = `± ${profile?.line_tolerance_px ?? 10} px`;
}
$('placeLine').onclick = async () => {
  if (!ready() || !await finishPendingRoad()) return;
  playing = false;
  setMode('line');
  status('Поставьте линию там, где автомобиль виден сбоку. Проезд определяется по центру рамки.');
};
$('removeLine').onclick = async () => {
  if (!ready() || !await finishPendingRoad()) return;
  checkpoint();profile.measurement_line_x = null;
  setMode('select');
  refresh();
};
$('lineTolerance').oninput = () => {
  if (!profile) return;
  if(sessionBusy){$('lineTolerance').value=profile.line_tolerance_px??10;return;}
  playing = false;
  checkpoint('line-tolerance');profile.line_tolerance_px = Number($('lineTolerance').value);
  updateLineLabel();
  draw();updateSaveState();
};
$('lineTolerance').onchange = () => { if (media) refresh(); };
updateLineLabel();

let stationSending = false;
async function sendToStation() {
  if (stationSending || !ready() || !await finishPendingRoad()) return;
  const d = detections[selected];
  if (!d) { status('Выберите автомобиль перед отправкой оператору.', true); return; }
  stationSending = true;
  $('sendStation').disabled = true;
  try {
    const response = await fetch('/api/station/capture', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({media_id:media.id,profile,frame:Number($('timeline').value),
        bbox:d.bbox,label:d.label,source:d.source??'manual',temporal:media.frames>1&&d.source!=='manual',actor:localStorage.getItem('ferryOperator')||'Оператор'})
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
    ruler.points.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,4*s,0,Math.PI*2);ctx.fill();ctx.fillText((i*ruler.step_m).toFixed(2)+' м',x+6*s,y-7*s);});
  }
}
function renderRulers(){
  $('rulers').replaceChildren();
  (profile?.metric_rulers||[]).forEach((r,i)=>{
    const row=document.createElement('div');row.className='ref';
    const text=document.createElement('span');text.textContent=`Линия ${i+1}: ${r.points.length-1} × ${r.step_m} m`;
    const remove=document.createElement('button');remove.textContent='Удалить';
    remove.onclick=()=>editProfile(next=>{next.metric_rulers.splice(i,1);});
    const spacing=document.createElement('input');spacing.type='number';spacing.min='.01';spacing.max='20';spacing.step='.01';spacing.value=r.step_m;spacing.ariaLabel=`Шаг мерной линии ${i+1}, м`;
    spacing.onchange=()=>editProfile(next=>{next.metric_rulers[i].step_m=Number(spacing.value);});
    row.append(text,spacing,remove);$('rulers').append(row);
  });
}
async function finishRuler(){
  if(mode!=='ruler')return true;
  if(draft.length<3){status('Добавьте минимум три последовательные мерные отметки. Точки сохранены на кадре.',true);return false;}
  const step=Number($('rulerStep').value);
  if(!Number.isFinite(step)||step<=0){status('Введите положительное расстояние между отметками.',true);return false;}
  const next=structuredClone(profile);next.metric_rulers??=[];
  next.metric_rulers.push({points:structuredClone(draft),step_m:step});
  try{await api('/validate-profile',next);checkpoint();profile=next;draft=[];setMode('select');renderReferences();return true;}
  catch(e){status(e.message,true);return false;}
}
$('drawRuler').onclick=async()=>{
  if(!ready()||!await finishPendingRoad())return;
  if(profile.polygon.length<4){status('Сначала отметьте дорогу.',true);return;}
  draft=[];setMode('ruler');status('Нажимайте на реальные мерные отметки по порядку вдоль дороги. Укажите измеренный шаг.');
};
$('finishRuler').onclick=async()=>{if(ready()&&await finishRuler())await refresh();};
$('undoRuler').onclick=()=>{if(mode==='ruler'){draft.pop();draw();}};
$('cancelRuler').onclick=()=>{if(mode==='ruler'){draft=[];setMode('select');}};

updateProfileInfo();

function detectorUI(){
  if(profile)detectorPreferences={detector_model:profile.detector_model??'yolo26n',detector_imgsz:profile.detector_imgsz??640};
  const {detector_model:name,detector_imgsz:size}=detectorPreferences;
  $('detectorModel').value=name;$('detectorSize').value=size;
  $('detectorModel').disabled=$('detectorSize').disabled=busy||sessionBusy;
  $('detect').textContent='Найти · '+detectorName(name);
  $('detectorStatus').textContent=busy||sessionBusy
    ?'Дождитесь завершения текущей операции — затем модель снова можно будет изменить.'
    :profile?'Выбрано для проверки кадра. Для работы станции нажмите «Сохранить и применить» сверху.'
    :'Загрузите сохранённую настройку или JSON, чтобы сменить модель без открытия кадра.';
  updateSaveState();
}
for(const id of ['detectorModel','detectorSize'])$(id).onchange=async()=>{
  if(sessionBusy||busy){detectorUI();status('Дождитесь завершения текущей операции, затем выберите модель.');return;}
  detectorPreferences={detector_model:$('detectorModel').value,detector_imgsz:Number($('detectorSize').value)};
  if(!profile){detectorUI();status('Модель выбрана. Откройте кадр или видео, чтобы начать новую калибровку.');return;}
  checkpoint();Object.assign(profile,detectorPreferences);
  revision++;detections=[];selected=-1;detectorUI();inspect();draw();
  status('Модель выбрана в черновике. Нажмите «Сохранить и применить», чтобы использовать её на станции. После смены модели проверьте калибровку.');
};
detectorUI();

function detectorName(name){return {'yolo26n':'YOLO26 N','yolo26m':'YOLO26 M','yolo26l':'YOLO26 L','rtdetr-l':'RT-DETR L','rtdetr-x':'RT-DETR X'}[name]||name;}
function profileSignature(value){return JSON.stringify(value);}
function updateSaveState(){
  const target=$('calibrationTarget').value==='operator'?'operator':'ip',saved=savedProfiles[target];
  const dirty=!!profile&&profileSignature(profile)!==profileSignature(saved);
  $('saveState').textContent=!profile?'Нет калибровки':dirty?'Есть неприменённые изменения':'Настройка сохранена';
  $('saveState').className=dirty?'dirty':'';
  $('serverDetector').textContent=saved?`На сервере: ${detectorName(saved.detector_model??'yolo26n')} · ${saved.detector_imgsz??640} px · ${target==='ip'?'IP-камеры':'видеозаписи'}`:'Сохранённая настройка ещё не загружена';
  $('applyCalibration').disabled=busy||sessionBusy||!profile;
  $('applyCalibration').textContent=sessionBusy?'Подождите…':'Сохранить и применить';
  $('calibrationTarget').disabled=busy||sessionBusy;
  $('undoChange').disabled=busy||!undoHistory.length;
}
function checkpoint(group=null){
  if(!profile)return;
  if(group&&group===undoGroup)return;
  const snapshot={profile:structuredClone(profile),media,picture,profileName,frame:Number($('timeline').value)};
  if(profileSignature(undoHistory.at(-1)?.profile)!==profileSignature(profile))undoHistory.push(snapshot);
  if(undoHistory.length>30)undoHistory.shift();
  undoGroup=group;updateSaveState();
}
$('undoChange').onclick=()=>sessionTask(async()=>{
  const previous=undoHistory.pop();if(!previous)return false;
  ({profile,media,profileName}=previous);undoGroup=null;pendingMedia=null;
  resetView();$('timeline').value=previous.frame;$('timeline').max=media?media.frames-1:0;
  $('filename').textContent=media?.name||'Кадр не открыт';$('empty').hidden=!!media;
  status('Предыдущая настройка восстановлена.');return !!media;
});
function measurementUI(){
  $('measurementMode').value=profile?.measurement_mode??'estimate';
  $('accuracyTolerance').value=Math.round((profile?.accuracy_tolerance_m??.1)*100);
}
$('measurementMode').onchange=async()=>{await editProfile(next=>{next.measurement_mode=$('measurementMode').value;});measurementUI();};
$('accuracyTolerance').onchange=async()=>{await editProfile(next=>{next.accuracy_tolerance_m=Number($('accuracyTolerance').value)/100;});measurementUI();};
function applyViewZoom(){
  const zoom=Number($('viewZoom').value)||1;
  canvas.style.width='';canvas.style.maxHeight='';$('stage').classList.remove('zoomed');
  if(zoom>1&&picture){
    const width=canvas.getBoundingClientRect().width*zoom,stage=$('stage');
    canvas.style.width=width+'px';stage.classList.add('zoomed');
    const points=profile?.polygon||[],center=points.length?points.reduce((sum,p)=>[sum[0]+p[0]/points.length,sum[1]+p[1]/points.length],[0,0]):[canvas.width/2,canvas.height/2];
    stage.scrollLeft=center[0]*width/canvas.width-stage.clientWidth/2;
    stage.scrollTop=center[1]*width/canvas.width-stage.clientHeight/2;
  }
}
$('viewZoom').onchange=applyViewZoom;
async function switchTarget(target){
  const previous=currentTarget;
  if(profile&&!await finishPendingRoad()){$('calibrationTarget').value=previous;return false;}
  sourceDrafts[previous]={profile,media,profileName,frame:Number($('timeline').value),saved:savedProfiles[previous]};
  currentTarget=target;undoHistory=[];undoGroup=null;profile=null;media=null;
  const draft=sourceDrafts[target];
  if(draft&&profileSignature(draft.profile)!==profileSignature(draft.saved)){({profile,media,profileName}=draft);resetView();$('timeline').value=draft.frame;$('timeline').max=media?media.frames-1:0;$('filename').textContent=media?.name||'Кадр не открыт';$('empty').hidden=!!media;return !!media;}
  resetView();$('filename').textContent='Кадр не открыт';$('empty').hidden=false;
  try{return await loadSavedProfile(target);}
  catch(e){status(e.message,true);return false;}
}
$('calibrationTarget').onchange=()=>sessionTask(()=>switchTarget($('calibrationTarget').value));
async function initializeCalibration(){
  if(window.parent!==window)document.body.classList.add('embedded');
  const requested=new URLSearchParams(location.search).get('target');
  const target=requested==='operator'||requested==='ip'?requested:localStorage.getItem('ferrySourceMode')==='video'?'operator':'ip';
  $('calibrationTarget').value=target;currentTarget=target;
  await sessionTask(async()=>{
    try{await loadSavedProfile(target);undoHistory=[];status('Настройка загружена. Модель можно сменить без открытия кадра. Для разметки получите кадр с камеры или откройте файл.');}
    catch(e){status('Сохранённая настройка недоступна: '+e.message+'. Импортируйте JSON или начните с кадра.',true);}
    return false;
  });
}
measurementUI();
if(typeof window!=='undefined'&&window.addEventListener){
  window.addEventListener('beforeunload',event=>{
    const target=$('calibrationTarget').value;
    const otherDirty=Object.entries(sourceDrafts).some(([key,draft])=>key!==target&&draft.profile&&profileSignature(draft.profile)!==profileSignature(draft.saved));
    if(otherDirty||(profile&&profileSignature(profile)!==profileSignature(savedProfiles[target]))){event.preventDefault();event.returnValue='';}
  });
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==window.parent||event.data?.type!=='station-settings-changed')return;
    // Never replace an unsaved draft when the operator changes the model elsewhere.
    if(profile&&profileSignature(profile)!==profileSignature(savedProfiles[$('calibrationTarget').value])){status('На станции изменили настройку. Ваш черновик сохранён; нажмите «Загрузить сохранённую», чтобы обновить его.');return;}
    if(event.data.target===$('calibrationTarget').value)$('loadSaved').onclick();
  });
  initializeCalibration();
}

function measurementStatus(value){return {pending:'Ожидание',outside_road:'Вне дороги',outside_calibration:'Вне зоны масштаба',waiting_for_line:'Ожидание линии',calibration_review:'Проверьте калибровку',depth_calibrated:'Масштаб по глубине',ruler_calibrated:'Мерные линии',constant_scale:'Постоянный масштаб',uncalibrated:'Нет масштаба',projective_estimate:'Оценка по проекции',estimated:'Приблизительная оценка',clipped:'Автомобиль обрезан кадром'}[value]||value||'Ожидание';}
