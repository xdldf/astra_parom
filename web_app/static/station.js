'use strict';
const $ = id => document.getElementById(id);
let categories={}, statuses=[], tariffRows=[], rows=[], current=null, mode='auto', page='operator';
let quoteSerial=0, quoteTimer, filterTimer, pending=false, dirty=false;
const plateLetters={A:'А',B:'В',C:'С',E:'Е',H:'Н',K:'К',M:'М',O:'О',P:'Р',T:'Т',X:'Х',Y:'У'};
const russianPlate = text => (text||'').toUpperCase().replace(/[ABCEHKMOPTXY]/g,c=>plateLetters[c]).replace(/\s/g,'');
const fmt = n => n == null ? 'Нужна проверка' : new Intl.NumberFormat('ru-RU').format(n)+' ₽';
const lengthText = n => n == null ? 'Не измерена' : new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n)+' м';
const statusClass = s => s==='Оплачен'||s==='Подтвержден'?'green':s==='Требует проверки'?'orange':s==='Отклонён'?'red':'blue';
function toast(text){$('toast').textContent=text;$('toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('show'),6000);}
async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body instanceof FormData?body:JSON.stringify(body)};
  const response=await fetch('/api/station'+path,options);
  const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
async function action(fn){if(pending)return;pending=true;try{await fn();}catch(e){toast(e.message);}finally{pending=false;}}
function options(id,entries,all){
  $(id).replaceChildren();
  if(all){let o=document.createElement('option');o.value='';o.textContent=all;$(id).append(o);}
  Object.entries(entries).forEach(([value,label])=>{let o=document.createElement('option');o.value=value;o.textContent=label;$(id).append(o);});
}
function cell(tr,text){let td=document.createElement('td');td.textContent=text;tr.append(td);return td;}
function badge(s){let span=document.createElement('span');span.className='tag '+statusClass(s);span.textContent=s;return span;}
function emptyTable(id,span,text){let tr=document.createElement('tr');tr.className='empty-row';let td=cell(tr,text);td.colSpan=span;$(id).append(tr);}
function query(report=false){
  const names=report?{date_from:'reportFrom',date_to:'reportTo',status:'reportStatus',category:'reportCategory'}:
    {plate:'filterPlate',status:'filterStatus',category:'filterCategory',length_min:'filterMin',length_max:'filterMax'};
  const params=new URLSearchParams();
  for(const [key,id] of Object.entries(names))if($(id).value.trim())params.set(key,$(id).value.trim());
  return params;
}
function tableRow(record,report=false){
  record=normalizeRecord(record);
  let tr=document.createElement('tr');
  if(!report){tr.classList.toggle('selected',current?.id===record.id);tr.onclick=()=>action(()=>selectRecord(record.id));}
  if(report)cell(tr,record.created_at.slice(0,10).split('-').reverse().join('.'));
  cell(tr,record.created_at.slice(11,19));cell(tr,record.plate||(record.plate_ocr?.candidates?.[0]?.text ? record.plate_ocr.candidates[0].text : record.plate_ocr?.state==='queued'?'Читаем номер…':'Не указан'));cell(tr,lengthText(record.length_m));
  cell(tr,categories[record.category]);cell(tr,fmt(record.tariff.amount_rub));cell(tr,'').append(badge(record.status));return tr;
}
async function loadVehicles(){
  const params=query();const result=await api('/vehicles?'+params);
  if(params.toString()!==query().toString())return;
  rows=result.rows;$('carsTable').replaceChildren();rows.forEach(r=>$('carsTable').append(tableRow(r)));
  if(!rows.length)emptyTable('carsTable',6,'Автомобилей пока нет или они не подходят под фильтры');
  $('totalCars').textContent=result.count;$('filterCount').textContent=params.size||'';
  $('updatedAt').textContent='Обновлено: '+new Date().toLocaleTimeString('ru-RU');
  if(current&&!dirty){const fresh=rows.find(r=>r.id===current.id);if(fresh&&fresh.version!==current.version){const id=current.id;const detail=await api('/vehicles/'+id);if(!dirty&&current?.id===id)renderRecord(detail);}}
}
function photo(id,emptyId,filename,version){
  $(id).hidden=!filename;$(emptyId).hidden=!!filename;
  if(filename)$(id).src='/api/station/photos/'+encodeURIComponent(filename)+'?v='+version;
  else $(id).removeAttribute('src');
}
function showQuote(q){
  $('tariff').textContent=fmt(q.amount_rub)+(q.code?' · п. '+q.code:'');
  $('confirm').textContent='✓ Подтвердить'+(q.amount_rub!=null?' • '+fmt(q.amount_rub):'');
  $('confirm').disabled=q.amount_rub==null||current?.status==='Оплачен';
  const notes=[...q.warnings];
  if(current?.source?.camera_note)notes.unshift(current.source.camera_note);
  if(current?.source)notes.unshift('Проверьте длину и категорию автомобиля.');
  $('warning').hidden=!notes.length;$('warningText').replaceChildren();
  notes.forEach(text=>{let small=document.createElement('small');small.textContent=text;$('warningText').append(small);});
  $('boundaryTariffs').replaceChildren();
  tariffRows.filter(t=>t.category===$('categoryInput').value).forEach(t=>{let d=document.createElement('div');d.textContent=`${t.code}: ${t.description} — ${fmt(t.amount_rub)}`;$('boundaryTariffs').append(d);});
}
function normalizeRecord(r){return {...r,plate:russianPlate(r.plate),plate_ocr:r.plate_ocr?{...r.plate_ocr,candidates:r.plate_ocr.candidates?.map(p=>({...p,text:russianPlate(p.text)}))}:null};}
function renderRecord(r){
  r=normalizeRecord(r);
  current=r;dirty=false;quoteSerial++;$('emptyDetail').hidden=true;$('recordDetail').hidden=false;
  renderOCR(r);
  $('plate').textContent=r.plate||'Не указан';$('category').textContent=categories[r.category];
  $('detectedLength').textContent=r.source?lengthText(r.measured_length_m):'Ручная запись';
  $('recordStatus').replaceChildren(badge(r.status));$('plateInput').value=r.plate;$('categoryInput').value=r.category;
  $('lengthInput').value=r.length_m??'';$('manualTariff').value=r.manual_rub??'';$('reason').value='';
  mode=r.manual_rub!=null?'manual':'auto';modeUI();
  photo('selectedFront','selectedFrontEmpty',r.front_photo,r.version);$('expandFront').disabled=!r.front_photo;
  photo('selectedMeasurement','selectedMeasurementEmpty',r.side_photo,r.version);$('expandMeasurement').disabled=!r.side_photo;$('selectedPhotoFrame').textContent=r.source?`Кадр ${r.source.frame}`:'';
  photo('frontImage','frontEmpty',r.front_photo,r.version);photo('sideImage','sideEmpty',r.side_photo,r.version);
  $('measureLabel').hidden=r.measured_length_m==null;$('measureLabel').textContent='≈ '+lengthText(r.measured_length_m);
  $('sourceLabel').textContent=r.source?`Кадр ${r.source.frame}`:'Ручная запись';
  const closed=r.status==='Оплачен';
  ['plateInput','categoryInput','lengthInput','reason','autoMode','manualMode','reject','edit','confirm','uploadFront'].forEach(id=>$(id).disabled=closed);
  $('manualTariff').disabled=closed||mode!=='manual';$('paid').disabled=r.status!=='Подтвержден';
  showQuote(r.tariff);$('history').replaceChildren();
  for(const h of r.history??[]){let d=document.createElement('div');d.className='history-row';d.textContent=`${h.timestamp.slice(0,19).replace('T',' ')} · ${h.actor==='OCR'?'Система':h.actor} · ${{plate_recognition:'Чтение номера',capture:'Измерение',edit:'Изменение',confirm:'Подтверждение',pay:'Оплата',create:'Создание'}[h.action]||h.action} · ${h.action==='plate_recognition'?'Номер прочитан по фото':h.reason||'—'}`;$('history').append(d);}
}
function renderOCR(r){
  const o=r.plate_ocr;
  $('recognizePlate').disabled=!r.front_photo||['Оплачен','Подтвержден'].includes(r.status)||o?.state==='queued';
  $('ocrStatus').textContent=!r.front_photo?'Нет снимка номера. Укажите номер вручную.':!o?'Нажмите «Распознать снова» для ранее сохранённого фото.':o.state==='queued'?'Читаем номер…':o.state==='error'?'Не удалось прочитать номер. Повторите попытку.':o.state==='not_found'?'Читаемый номер не найден. Можно указать его вручную.':'Найдены варианты. Проверьте символы и соответствие измеренному автомобилю.';
  $('ocrCandidates').replaceChildren();
  for(const p of o?.candidates||[]){
    const card=document.createElement('div');card.className='ocr-card';
    const picture=document.createElement('button');picture.className='ocr-picture';picture.title='Увеличить номер';
    const img=document.createElement('img');img.src='/api/station/photos/'+encodeURIComponent(p.photo);img.alt='Фото номера '+p.text;picture.append(img);
    picture.onclick=()=>{$('expandedMeasurement').src=img.src;$('measurementCaption').textContent=p.text;$('measurementDialog').showModal();};
    const label=document.createElement('strong');label.textContent=p.text;
    const info=document.createElement('small');info.textContent=p.votes>1?'Проверен по нескольким снимкам':'Проверьте номер по фото';
    const use=document.createElement('button');use.className='small-btn';use.textContent='Использовать номер';use.disabled=['Оплачен','Подтвержден'].includes(r.status);
    use.onclick=()=>{$('plateInput').value=p.text;$('reason').value='Уточнение номера по фото';changed();};
    card.append(picture,label,info,use);$('ocrCandidates').append(card);
  }
}
$('recognizePlate').onclick=()=>action(async()=>{if(dirty)throw Error('Сначала сохраните изменения');renderRecord(await api('/vehicles/'+current.id+'/recognize-plate',{}));});
async function selectRecord(id){
  if(dirty){toast('Сохраните изменения текущего автомобиля перед выбором другого.');return;}
  const record=await api('/vehicles/'+id);renderRecord(record);await loadVehicles();
}
function fields(){
  const len=$('lengthInput').value.trim();const manual=$('manualTariff').value.trim();
  if(mode==='manual'&&!manual)throw Error('Укажите ручной тариф');
  return {plate:$('plateInput').value,category:$('categoryInput').value,length_m:len?Number(len):null,
    manual_rub:mode==='manual'?Number(manual):null,actor:$('actor').value.trim()||'Оператор',reason:$('reason').value.trim()};
}
async function recalc(){
  if(!current)return;const serial=++quoteSerial;$('confirm').disabled=true;
  try{const q=await api('/quote',fields());if(serial===quoteSerial)showQuote(q);}catch(e){if(serial===quoteSerial){$('tariff').textContent=e.message;$('confirm').disabled=true;}}
}
function changed(){dirty=true;clearTimeout(quoteTimer);quoteSerial++;$('confirm').disabled=true;quoteTimer=setTimeout(recalc,150);}
function modeUI(){$('autoMode').classList.toggle('active',mode==='auto');$('manualMode').classList.toggle('active',mode==='manual');$('manualTariff').disabled=mode!=='manual';}
async function mutate(kind){
  if(!current)return;
  if(kind==='pay'&&dirty)throw Error('Перед отметкой оплаты сохраните и подтвердите изменения.');
  const r=await api('/vehicles/'+current.id,{...fields(),version:current.version,action:kind});
  renderRecord(await api('/vehicles/'+r.id));await loadVehicles();await syncClient();
  toast({edit:'Изменения сохранены. Запись требует подтверждения.',confirm:'Автомобиль подтверждён и показан клиенту.',reject:'Автомобиль отклонён.',pay:'Полученная оплата отмечена.'}[kind]);
}
async function syncClient(){
  const {vehicle:r}=await api('/client');$('clientEmpty').hidden=!!r;$('clientData').hidden=!r;if(!r)return;
  $('clientConfirmed').textContent=r.status==='Оплачен'?'✓ Оплата отмечена оператором':'✓ Данные подтверждены оператором';
  $('clientPlate').textContent=russianPlate(r.plate)||'Гос. номер не указан';$('clientCategory').textContent=categories[r.category];
  $('clientLength').textContent='📏 Длина: '+lengthText(r.length_m);$('clientPrice').textContent=fmt(r.tariff.amount_rub);
  $('clientPriceLabel').textContent=r.status==='Оплачен'?'Оплачено':'К оплате';
  photo('clientFront','clientFrontEmpty',r.front_photo,r.version);photo('clientSide','clientSideEmpty',r.side_photo,r.version);
}
async function report(){
  const params=query(true);const data=await api('/vehicles?'+params);
  if(params.toString()!==query(true).toString())return;
  $('reportTable').replaceChildren();data.rows.forEach(r=>$('reportTable').append(tableRow(r,true)));
  if(!data.rows.length)emptyTable('reportTable',7,'За выбранный период записей нет');
  $('reportCount').textContent=data.count;$('reportAmount').textContent=fmt(data.amount_rub);
  $('reportPaid').textContent='Из них оплачено: '+fmt(data.paid_rub);$('reportIssues').textContent=data.issues;
}
function showPage(name){
  page=name;document.querySelectorAll('.page').forEach(el=>el.classList.toggle('active',el.id===name));
  document.querySelectorAll('[data-page]').forEach(el=>el.classList.toggle('active',el.dataset.page===name));
  if(name==='client')action(syncClient);if(name==='reports')action(report);
  if(name==='operator')action(loadVehicles);
  if(name==='calibration'&&!$('calibrationFrame').getAttribute('src'))$('calibrationFrame').src=$('calibrationFrame').dataset.src;
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>showPage(b.dataset.page));
$('filterBtn').onclick=()=>$('filterPanel').classList.toggle('open');$('refresh').onclick=()=>action(loadVehicles);
['filterPlate','filterStatus','filterCategory','filterMin','filterMax'].forEach(id=>$(id).oninput=()=>{clearTimeout(filterTimer);filterTimer=setTimeout(()=>action(loadVehicles),200);});
$('resetFilters').onclick=()=>{['filterPlate','filterStatus','filterCategory','filterMin','filterMax'].forEach(id=>$(id).value='');action(loadVehicles);};
['plateInput','categoryInput','lengthInput','manualTariff'].forEach(id=>$(id).oninput=changed);
$('reason').oninput=()=>{dirty=true;};
$('autoMode').onclick=()=>{mode='auto';modeUI();changed();};$('manualMode').onclick=()=>{mode='manual';modeUI();changed();};
['edit','confirm','reject'].forEach(kind=>$(kind).onclick=()=>action(()=>mutate(kind)));
$('paid').onclick=()=>action(()=>mutate('pay'));
$('historyToggle').onclick=()=>$('history').hidden=!$('history').hidden;
$('discardEdit').onclick=()=>action(async()=>{if(!current)return;renderRecord(await api('/vehicles/'+current.id));await loadVehicles();});
$('newRecord').onclick=()=>action(async()=>{
  if(dirty)throw Error('Сначала сохраните изменения текущего автомобиля.');
  const r=await api('/vehicles',{actor:$('actor').value.trim()||'Оператор'});renderRecord(r);await loadVehicles();
});
$('uploadFront').onclick=()=>$('frontFile').click();
$('frontFile').onchange=e=>action(async()=>{
  const file=e.target.files[0];if(!file||!current)return;
  if(dirty)throw Error('Сначала сохраните изменения текущего автомобиля.');
  const fd=new FormData();fd.append('file',file);fd.append('version',current.version);fd.append('actor',$('actor').value);
  const r=await api('/vehicles/'+current.id+'/front-photo',fd);renderRecord(await api('/vehicles/'+r.id));e.target.value='';toast('Фото сохранено');
});
['reportFrom','reportTo','reportStatus','reportCategory'].forEach(id=>$(id).onchange=()=>action(report));
$('resetReport').onclick=()=>{['reportFrom','reportTo','reportStatus','reportCategory'].forEach(id=>$(id).value='');action(report);};
$('exportCSV').onclick=()=>action(async()=>{
  await report();const r=await fetch('/api/station/reports.csv?'+query(true));if(!r.ok)throw Error('Не удалось выгрузить отчёт');
  const url=URL.createObjectURL(await r.blob());let a=document.createElement('a');a.href=url;a.download='ferry_report.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$('actor').value=localStorage.getItem('ferryOperator')||'Оператор';$('actor').onchange=()=>localStorage.setItem('ferryOperator',$('actor').value);
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===$('calibrationFrame').contentWindow&&e.data?.type==='station-capture'){action(loadVehicles);toast('Автомобиль добавлен в очередь оператора');}});
async function start(){
  const data=await api('/catalog');categories=data.categories;statuses=data.statuses;tariffRows=data.tariffs;
  options('categoryInput',categories);['filterCategory','reportCategory'].forEach(id=>options(id,categories,'Все категории'));
  ['filterStatus','reportStatus'].forEach(id=>options(id,Object.fromEntries(statuses.map(s=>[s,s])),'Все статусы'));
  $('tariffPolicy').textContent=data.policy;tariffRows.forEach(t=>{let tr=document.createElement('tr');[t.code,t.category_label,t.description,fmt(t.amount_rub)].forEach(v=>cell(tr,v));$('tariffTable').append(tr);});
  const health=await api('/health');$('systemStatus').textContent=health.gpu_available?'● Система готова':'⚠ Система недоступна';$('systemStatus').title=health.gpu??'';
  const initial=new URLSearchParams(location.search).get('page');
  if(initial==='client'){document.body.classList.add('client-only');showPage('client');}else{await loadVehicles();}
  setInterval(async()=>{if(pending||document.hidden)return;try{if(page==='client')await syncClient();else if(page==='operator')await loadVehicles();}catch(e){$('systemStatus').textContent='⚠ Нет связи с сервером';}},3000);
}
start().catch(e=>{toast(e.message);$('systemStatus').textContent='⚠ Ошибка подключения';});
let ipMode=false,ipRunning=false,ipPollTimer=null;
let liveSource=null, liveGeneration=0, liveSnapshot=null, importedProfile=null;
let liveTracks=new LiveTracks(), streamId=null, streamFrame=0, streamTimer=null, streamStarting=false;
function calibrationLabel(){const p=liveSource?.profile||importedProfile;$('calibrationStatus').textContent=p?.references?.length?`Калибровка активна · ${p.references.length} эталонов · ${p.measurement_line_x==null?'вся дорога':'измерение у линии'}`:'Импортируйте JSON с дорогой и эталонными длинами';}
async function workbench(path,body){const response=await fetch('/api/workbench'+path,body instanceof FormData?{method:'POST',body}:body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
async function stopStream(){const id=streamId;streamId=null;liveGeneration++;clearTimeout(streamTimer);$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('streamPlay').textContent='▶ Пуск';if(id)await fetch('/api/stream/'+id,{method:'DELETE'});}
async function startStream(){
 if(streamStarting||streamId)return;if(!liveSource){toast('Откройте видео');return;}streamStarting=true;
 try{const response=await fetch('/api/stream/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:liveSource.media.id,profile:liveSource.profile,frame:streamFrame,front_media_id:liveSource.frontMedia?.id,front_offset_seconds:liveSource.frontOffset??3})});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));streamId=data.id;liveGeneration++;liveTracks=new LiveTracks();liveSnapshot=null;$('operatorDetection').src='/api/stream/'+streamId+'/video';if(liveSource.frontMedia)$('frontStream').src='/api/stream/'+streamId+'/front';$('streamPlay').textContent='Ⅱ Пауза';pollStream(streamId,liveGeneration);}
 catch(e){toast(e.message);}finally{streamStarting=false;}
}
async function pollStream(id,generation){
 if(id!==streamId||generation!==liveGeneration)return;
 try{const response=await fetch('/api/stream/'+id+'/state');if(!response.ok)throw Error('Поток завершён. Нажмите Пуск для повторного запуска.');const state=await response.json();if(id!==streamId||generation!==liveGeneration)return;
 streamFrame=state.frame;$('frontStatus').textContent=state.front_seconds==null?'Не подключена':state.plate_error?'Номер временно недоступен':(state.plates?.candidates?.map(p=>p.text).join(', ')||'Камера работает');$('syncStatus').textContent=state.sync_error_ms==null?'Одна камера':'Камеры синхронизированы';$('streamSeek').value=100*streamFrame/(liveSource.media.frames-1);
 $('detectionStatus').textContent=state.error?'Не удалось обработать изображение':'Измерение работает';
 const result=state.result,source=structuredClone(liveSource);
 if(result&&result.frame!==liveSnapshot?.frame){const frame=result.frame;
 liveSnapshot={source,frame,detections:result.detections};
 $('liveCar').replaceChildren();result.detections.forEach((d,i)=>{const option=document.createElement('option');option.value=i;option.textContent=(i+1)+': '+d.label+' · '+(d.length_m==null?'длина не измерена':d.length_m.toFixed(2)+' м');$('liveCar').append(option);});$('captureLive').disabled=!result.detections.length;
      const tracks=liveTracks.update(result.detections,frame/source.media.fps);
      if($('autoMeasure').checked){for(let i=0;i<result.detections.length;i++){const d=result.detections[i],t=tracks[i];if(t.sent)continue;
        if(d.length_m!=null){t.sent=true;queueLive({source,frame},d).catch(e=>{t.sent=false;toast(e.message);});continue;}
        const line=source.profile.measurement_line_x,prev=t.previous;
        if(line!=null&&prev&&d.depth!=null){const before=prev.box[0]+prev.box[2]/2-line,after=d.bbox[0]+d.bbox[2]/2-line;
          if(before*after<0&&frame/source.media.fps-prev.time<2){t.sent=true;const crossFrame=Math.round((prev.time+(frame/source.media.fps-prev.time)*Math.abs(before)/(Math.abs(before)+Math.abs(after)))*source.media.fps);
          captureCrossing(source,crossFrame,d).then(ok=>{if(!ok)t.sent=false;}).catch(e=>{t.sent=false;toast(e.message);});}}
        }}

 }
 }catch(e){if(id===streamId){$('detectionStatus').textContent=e.message;await stopStream();}return;}
 if(id===streamId)streamTimer=setTimeout(()=>pollStream(id,generation),100);
}
async function openLive(source,frame=0){await stopStream();liveSource=structuredClone(source);streamFrame=frame;calibrationLabel();localStorage.setItem('ferryVideo',JSON.stringify(source));$('liveStatus').textContent=source.media.name;showPage('operator');await startStream();}
$('streamPlay').onclick=()=>ipMode?toggleIp():streamId?stopStream():startStream();
$('streamSeek').onchange=async()=>{if(!liveSource)return;const frame=Math.round(Number($('streamSeek').value)*(liveSource.media.frames-1)/100),running=!!streamId;await stopStream();streamFrame=frame;if(running)await startStream();};
$('operatorFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{$('liveStatus').textContent='Загрузка видео…';const fd=new FormData();fd.append('file',file);const media=await workbench('/media',fd);const previous=importedProfile||liveSource?.profile;if(previous&&previous.image_size.toString()!==media.image_size.toString())throw Error('Разрешение видео не совпадает с калибровкой');const profile=previous||{version:1,image_size:media.image_size,lens:{},polygon:[],references:[]};await openLive({media,profile,frontMedia:liveSource?.frontMedia,frontOffset:liveSource?.frontOffset??3});}catch(error){toast(error.message);}e.target.value='';};
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===$('calibrationFrame').contentWindow&&e.data?.type==='station-play')openLive({media:e.data.media,profile:e.data.profile,frontMedia:liveSource?.frontMedia,frontOffset:liveSource?.frontOffset??3},e.data.frame).catch(e=>toast(e.message));});
$('captureLive').onclick=()=>action(async()=>{const snapshot=structuredClone(liveSnapshot),d=snapshot?.detections[Number($('liveCar').value)];if(d)await queueLive(snapshot,d);});
try{const saved=JSON.parse(localStorage.getItem('ferryVideo'));if(saved?.media&&saved?.profile){liveSource=saved;calibrationLabel();$('liveStatus').textContent=saved.media.name;}}catch{}
window.addEventListener('pagehide',()=>{if(streamId)fetch('/api/stream/'+streamId,{method:'DELETE',keepalive:true});});
async function queueLive(snapshot,d){
  const record=await api('/capture',{media_id:snapshot.source.media.id,profile:snapshot.source.profile,frame:snapshot.frame,bbox:d.bbox,label:d.label,source:'yolo26n',actor:$('actor').value||'Оператор',front_media_id:snapshot.source.frontMedia?.id,front_offset_seconds:snapshot.source.frontOffset??3});
  await loadVehicles();if(!dirty&&!current)renderRecord(await api('/vehicles/'+record.id));return record;
}
$('operatorCalibration').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{
  const parsed=JSON.parse(await file.text());const profile=await workbench('/validate-profile',parsed.profile||parsed);
  if(!ipMode&&liveSource&&profile.image_size.toString()!==liveSource.media.image_size.toString())throw Error('Разрешение калибровки не совпадает с видео');
  if(ipMode){await ipApi('/stop',{});ipRunning=false;await ipApi('/calibration',profile);$('calibrationStatus').textContent='Настройка камер сохранена';toast('Настройка загружена. Подключите камеры снова.');e.target.value='';return;}
  await workbench('/operator-calibration',profile);importedProfile=profile;
  if(ipMode&&ipRunning)await ipApi('/stop',{});
  if(!ipMode&&liveSource){const running=!!streamId;await stopStream();liveSource.profile=profile;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));if(running)await startStream();}
  calibrationLabel();toast('Калибровка загружена. Измерение включено.');$('autoMeasure').checked=true;
}catch(error){toast(error.message);}e.target.value='';};
workbench('/operator-calibration').then(profile=>{importedProfile=profile;if(liveSource&&profile.image_size.toString()===liveSource.media.image_size.toString()){liveSource.profile=profile;liveGeneration++;}calibrationLabel();}).catch(()=>calibrationLabel());

async function captureCrossing(source,frame,original){
  const result=await workbench('/frame/'+source.media.id,{profile:source.profile,frame,detect:true,confidence:.3,boxes:[]});
  const eligible=result.detections.filter(d=>d.length_m!=null&&Math.abs(d.bottom[1]-original.bottom[1])<original.bbox[3]);
  eligible.sort((a,b)=>Math.abs(a.bottom[1]-original.bottom[1])-Math.abs(b.bottom[1]-original.bottom[1]));
  if(!eligible.length)return false;await queueLive({source,frame},eligible[0]);return true;
}

$('expandMeasurement').onclick=()=>{if(!current?.side_photo)return;$('expandedMeasurement').src=$('selectedMeasurement').src;$('measurementCaption').textContent=`${current.plate||'Номер не указан'} · ${lengthText(current.measured_length_m)}${current.source?' · кадр '+current.source.frame:''}`;$('measurementDialog').showModal();};
$('closeMeasurement').onclick=()=>$('measurementDialog').close();
$('measurementDialog').onclick=e=>{if(e.target!==$('measurementDialog'))return;const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();};

$('expandFront').onclick=()=>{if(!current?.front_photo)return;$('expandedMeasurement').src=$('selectedFront').src;const p=current.source?.front_camera;$('measurementCaption').textContent=p?.kind==='ip'?'Снимки камер в момент измерения. Проверьте соответствие автомобиля.':p?`Синхронный кадр спереди ${p.frame} · ${p.front_seconds.toFixed(2)} с · боковой ${p.side_seconds.toFixed(2)} с. Проверьте соответствие автомобиля.`:'Фото спереди';$('measurementDialog').showModal();};
$('frontOffset').oninput=()=>{$('frontOffsetValue').textContent=(Number($('frontOffset').value)>=0?'+':'')+Number($('frontOffset').value).toFixed(2)+' с';};
$('frontOffset').onchange=async()=>{if(!liveSource)return;const running=!!streamId;await stopStream();liveSource.frontOffset=Number($('frontOffset').value);localStorage.setItem('ferryVideo',JSON.stringify(liveSource));if(running)await startStream();};
$('frontVideoFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const fd=new FormData();fd.append('file',file);const frontMedia=await workbench('/media',fd);if(frontMedia.frames<2)throw Error('Выберите видео');const running=!!streamId;await stopStream();if(!liveSource)throw Error('Сначала выберите боковое видео');liveSource.frontMedia=frontMedia;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));$('frontStatus').textContent=frontMedia.name;if(running)await startStream();}catch(e){toast(e.message);}e.target.value='';};
$('streamPlay').disabled=true;
fetch('/api/stream/configuration').then(async response=>{if(!response.ok)return;const pair=await response.json();if(!liveSource?.frontMedia){liveSource=pair;localStorage.setItem('ferryVideo',JSON.stringify(pair));}else if(importedProfile)liveSource.profile=importedProfile;calibrationLabel();$('liveStatus').textContent=liveSource.media.name;$('frontStatus').textContent=liveSource.frontMedia.name;$('frontOffset').value=liveSource.frontOffset??3;$('frontOffset').oninput();}).catch(e=>toast(e.message)).finally(()=>{$('streamPlay').disabled=false;});

$('plateInput').addEventListener('input',()=>{const field=$('plateInput'),pos=field.selectionStart;field.value=russianPlate(field.value);field.setSelectionRange(pos,pos);});

async function ipApi(path,body){
 const r=await fetch('/api/ip'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Проверьте настройки камер');return data;
}
async function ipPoll(){
 clearTimeout(ipPollTimer);if(!ipMode)return;
 try{const s=await ipApi('/state');if(!ipMode)return;const wasRunning=ipRunning;ipRunning=s.running;
 $('streamPlay').textContent=s.running?'■ Остановить камеры':'▶ Подключить камеры';
 $('ipStatus').textContent=!s.running?'Камеры остановлены':s.error||(!s.side_ready&&!s.front_ready?'Нет связи с камерами':!s.front_ready?'Боковая камера работает · фронтальная недоступна':!s.side_ready?'Фронтальная камера работает · боковая недоступна':'Камеры работают');
 if(s.running){
  if(!wasRunning||!$('operatorDetection').getAttribute('src')){$('operatorDetection').src='/api/ip/side/video';$('frontStream').src='/api/ip/front/video';}
  $('operatorDetection').hidden=!s.side_ready;$('frontStream').hidden=!s.front_ready;
  $('liveStatus').textContent=s.side;$('frontStatus').textContent=s.front_ready?(s.plates?.candidates?.map(p=>p.text).join(', ')||s.front):s.front;
  $('syncStatus').textContent=s.ready?'Изображения согласованы':s.side_ready?'Длина измеряется без фронтального снимка':s.front_ready?'Номера читаются · длина недоступна':'Ожидание камер';
  $('detectionStatus').textContent=s.error||(!s.side_ready?'Ожидание боковой камеры':s.auto_measure?'Автоматическое измерение включено':'Автоматическое измерение выключено');
  $('autoMeasure').checked=s.auto_measure;
 }else{$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('liveStatus').textContent='Не подключена';$('frontStatus').textContent='Не подключена';$('syncStatus').textContent='Ожидание подключения';$('detectionStatus').textContent='Камеры остановлены';}
 }catch(e){$('ipStatus').textContent='Нет связи с программой';}
 if(ipMode)ipPollTimer=setTimeout(ipPoll,500);
}
async function toggleIp(){
 $('streamPlay').disabled=true;
 try{await ipApi(ipRunning?'/stop':'/start',{});await ipPoll();}catch(e){toast(e.message);}finally{$('streamPlay').disabled=false;}
}
$('sourceMode').onchange=()=>action(async()=>{
 const next=$('sourceMode').value==='ip';
 if(next){await stopStream();ipMode=true;}else{await ipApi('/stop',{});ipRunning=false;ipMode=false;clearTimeout(ipPollTimer);$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('streamPlay').textContent='▶ Пуск';}
 document.body.classList.toggle('ip-mode',ipMode);$('ipSettings').hidden=!ipMode;
 $('operatorFile').parentElement.hidden=ipMode;$('frontVideoFile').parentElement.hidden=ipMode;$('frontOffset').parentElement.hidden=ipMode;
 $('operatorDetection').hidden=false;$('frontStream').hidden=false;
 localStorage.setItem('ferrySourceMode',ipMode?'ip':'video');
 if(ipMode){const cfg=await ipApi('/configuration');$('calibrationStatus').textContent=cfg.calibrated?'Настройка камер сохранена':'Используется общая настройка; при необходимости загрузите настройку камер';await ipPoll();}else{calibrationLabel();$('ipStatus').textContent='';$('liveStatus').textContent=liveSource?.media.name||'Откройте видео';$('frontStatus').textContent=liveSource?.frontMedia?.name||'Не подключена';$('syncStatus').textContent='Общий таймер двух камер';$('detectionStatus').textContent='Ожидание видео';}
});
$('saveIp').onclick=()=>action(async()=>{
 const cfg=await ipApi('/configuration',{side_url:$('ipSideUrl').value.trim(),front_url:$('ipFrontUrl').value.trim(),offset_seconds:Number($('ipOffset').value),tolerance_ms:Number($('ipTolerance').value),auto_measure:$('autoMeasure').checked,autostart:$('ipAutostart').checked});
 $('ipSideUrl').value='';$('ipFrontUrl').value='';$('ipSideUrl').placeholder=cfg.side_configured?'Адрес сохранён':'rtsp://…';$('ipFrontUrl').placeholder=cfg.front_configured?'Адрес сохранён':'rtsp://…';toast('Подключение сохранено');
});
$('autoMeasure').onchange=()=>{if(ipMode)ipApi('/automatic?enabled='+$('autoMeasure').checked,{}).catch(e=>toast(e.message));};
ipApi('/configuration').then(async cfg=>{
 $('ipOffset').value=cfg.offset_seconds;$('ipTolerance').value=cfg.tolerance_ms;$('ipAutostart').checked=cfg.autostart;
 $('ipSideUrl').placeholder=cfg.side_configured?'Адрес сохранён':'rtsp://…';$('ipFrontUrl').placeholder=cfg.front_configured?'Адрес сохранён':'rtsp://…';
 const state=await ipApi('/state');if(state.running||localStorage.getItem('ferrySourceMode')==='ip'){$('sourceMode').value='ip';$('sourceMode').onchange();}
}).catch(e=>toast(e.message));
