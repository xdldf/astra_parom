'use strict';
const $ = id => document.getElementById(id);
let categories={}, statuses=[], tariffRows=[], rows=[], current=null, mode='auto', page='operator';
let referenceDirty=false;
let quoteSerial=0, quoteTimer, filterTimer, pending=false, dirty=false;
let queueOffset=0, queueSnapshot=null, queuePage=null, reportOffset=0, reportSnapshot=null, reportPage=null;
let queueSerial=0, reportSerial=0, pollBusy=false, ipSessionId=null;
let tariffsEnabled=true;
let clientPageSize=20;
let clientOffset=0, clientSnapshot=null, clientPage=null, clientSerial=0;
function clientConfigurationPath(){return '/client-configuration?station_id='+encodeURIComponent($('stationId').value);}
function applyClientConfiguration(config){
  if(clientPageSize!==config.rows_per_page){clientOffset=0;clientSnapshot=null;clientPage=null;}
  clientPageSize=config.rows_per_page;
  $('clientRowsPerPage').value=String(clientPageSize);
  $('clientRowsPerPage').disabled=false;
  $('client').style.setProperty('--client-page-size',clientPageSize);
  $('client').classList.toggle('client-dense',clientPageSize>=10);
}
$('clientRowsPerPage').onchange=()=>action(async()=>{
  const input=$('clientRowsPerPage'),path=clientConfigurationPath();input.disabled=true;clientSerial++;
  try{
    const config=await api(path,{rows_per_page:Number(input.value)});
    getCache.delete(path);applyClientConfiguration(config);await syncClient();
  }finally{input.value=String(clientPageSize);input.disabled=false;}
});
function operatorPrices(visible){
  document.body.classList.toggle('operator-prices-hidden',!visible);
  $('showOperatorPrices').checked=visible;
  document.querySelectorAll('#carsTable .empty-row td').forEach(td=>td.colSpan=tariffsEnabled&&visible?7:6);
}
operatorPrices(localStorage.getItem('ferryOperatorPrices')!=='false');
$('showOperatorPrices').onchange=()=>{
  const visible=$('showOperatorPrices').checked;
  localStorage.setItem('ferryOperatorPrices',String(visible));operatorPrices(visible);
};
const getCache=new Map();
async function cachedGet(path){
  const old=getCache.get(path);
  const response=await fetch('/api/station'+path,{cache:'no-store',headers:old?{'If-None-Match':old.etag}:{}});
  if(response.status===304)return old.data;
  const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  getCache.delete(path);getCache.set(path,{etag:response.headers.get('ETag'),data});
  if(getCache.size>12)getCache.delete(getCache.keys().next().value);
  return data;
}
function pageQuery(report=false){
  const params=query(report),offset=report?reportOffset:queueOffset,snapshot=report?reportSnapshot:queueSnapshot;
  params.set('limit','250');params.set('offset',offset);
  if(offset&&snapshot!==null)params.set('snapshot',snapshot);
  return params;
}
function pagination(data,prefix){
  $(prefix+'Prev').disabled=data.offset===0;$(prefix+'Next').disabled=!data.has_more;
  $(prefix+'Range').textContent=data.count?`${data.offset+1}–${data.offset+data.rows.length} из ${data.count}`:'0';
}
function applyTariffMode(enabled){
  if(typeof enabled!=='boolean')return;
  const changed=tariffsEnabled!==enabled;tariffsEnabled=enabled;
  document.body.classList.toggle('tariffs-off',!enabled);$('tariffsEnabled').checked=enabled;
  $('tariffModeNote').textContent=enabled?'Тарифы включены для всех операторов':'Только измерение · тарифы отключены для всех операторов';
  if(changed){
    quoteSerial++;clearTimeout(quoteTimer);
    if(current){showQuote(current.tariff);if(enabled&&current.status!=='Оплачен')recalc();}
  }
}
async function syncConfiguration(){applyTariffMode((await cachedGet('/configuration')).tariffs_enabled);$('tariffsEnabled').disabled=false;}
$('tariffsEnabled').onchange=()=>{
  if(pending){$('tariffsEnabled').checked=tariffsEnabled;return;}
  return action(async()=>{
  const enabled=$('tariffsEnabled').checked;$('tariffsEnabled').disabled=true;
  try{const cfg=await api('/configuration',{tariffs_enabled:enabled});applyTariffMode(cfg.tariffs_enabled);toast(enabled?'Тарифы включены':'Только измерение: тарифы отключены');}
  finally{$('tariffsEnabled').checked=tariffsEnabled;$('tariffsEnabled').disabled=false;}
  });
};

const plateLetters={A:'А',B:'В',C:'С',E:'Е',H:'Н',K:'К',M:'М',O:'О',P:'Р',T:'Т',X:'Х',Y:'У'};
const russianPlate = text => (text||'').toUpperCase().replace(/[ABCEHKMOPTXY]/g,c=>plateLetters[c]).replace(/\s/g,'');
const fmt = n => n == null ? 'Нужна проверка' : new Intl.NumberFormat('ru-RU').format(n)+' ₽';
const lengthText = n => n == null ? 'Не измерена' : new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n)+' м';
const approximateMeasurement = record => !!(record.source?.measurement?.approximate??record.measurement_approximate);
const recordLengthText = record => (record.length_m!=null&&approximateMeasurement(record)?'≈ ':'')+lengthText(record.length_m);
function measurementNote(record){
  const approximate=record.length_m!=null&&approximateMeasurement(record);
  if(record.length_m!=null&&!approximate)return '';
  const m=record.source?.measurement;
  const reasons=m?.quality_reasons?.length?m.quality_reasons:m?.temporal?.reasons?.length?m.temporal.reasons:[record.measurement_reason||m?.status];
  const texts={
    insufficient_temporal_frames:'Недостаточно кадров автомобиля у линии.',
    missing_line_evidence:'Нет пригодного кадра на линии измерения.',
    missed_measurement_line:'Центр автомобиля не попал на линию. Полный кадр сохранён для проверки.',
    capture_review:'Автомобиль сохранён для проверки без надёжной длины.',
    line_not_bracketed:'Не хватает кадров до и после линии.',
    ambiguous_vehicle_association:'Рамки автомобилей пересекаются: нужно проверить проезд.',
    unstable_temporal_length:'Длина по кадрам расходится больше допуска.',
    anchor_outside_calibration:'Автомобиль вне области калибровки.',
    outside_calibration:'Автомобиль вне области калибровки.',
    calibration_review:'Калибровка не прошла проверку геометрии.',
    needs_reference:'В настройке камеры нет эталонов длины.',
    outside_road:'Рамка автомобиля вне полигона дороги.',
    clipped:'Автомобиль не целиком в кадре.',
    catalogue_wheel_recovery_estimate:'Резервная оценка по контуру и колёсам; модель обучена на разных положениях автомобиля. Требуется проверка.',
    wheel_recovery_unavailable:'Резервная оценка не прошла проверку; длина не назначена, полный кадр сохранён.',
    catalogue_wheel_estimate:'Оценка по контуру кузова и видимым колёсам; каталожное сравнение не гарантирует точность 10 см.',
    wheel_refinement_unavailable:'Колёса не дали надёжного уточнения; сохранена оценка контура кузова.',
    catalogue_outline_estimate:'Оценка по контуру кузова и каталожным эталонам; точность 10 см для каждого автомобиля пока не достигнута.',
    outline_unavailable:'Контурная оценка недоступна. Полный кадр сохранён, длину нужно проверить.'
  };
  const note=[...new Set(reasons.map(r=>r==='road_overlap_proxy'?'Положение взято по пересечению рамки с дорогой.':texts[r]).filter(Boolean))].join(' ');
  return (approximate?'Приблизительно · требуется проверка. ':'')+note;
}
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
function emptyTable(id,span,text){let tr=document.createElement('tr');tr.className='empty-row';let td=cell(tr,text);td.colSpan=span-(!tariffsEnabled||(id==='carsTable'&&!$('showOperatorPrices').checked)?1:0);$(id).append(tr);}
function query(report=false){
  const names=report?{date_from:'reportFrom',date_to:'reportTo',status:'reportStatus',category:'reportCategory'}:
    {plate:'filterPlate',status:'filterStatus',category:'filterCategory',length_min:'filterMin',length_max:'filterMax'};
  const params=new URLSearchParams();
  for(const [key,id] of Object.entries(names))if($(id).value.trim())params.set(key,$(id).value.trim());
  return params;
}
function frontPhotoNote(record){
  const offset=record.front_photo_offset_seconds??record.source?.front_evidence?.offset_seconds;
  return offset==null?'Фото спереди':`Фото номера · ${Math.abs(offset).toFixed(1)} с ${offset<=0?'до':'после'} синхронного кадра. Проверьте соответствие автомобиля.`;
}
function carPhotos(tr,record){
  const td=cell(tr,'');td.className='car-photos';
  for(const [field,label] of [['side_photo','Фото сбоку'],['front_photo','Фото спереди']]){
    if(!record[field])continue;
    const url='/api/station/photos/'+encodeURIComponent(record[field])+'?v='+record.version;
    const button=document.createElement('button');button.type='button';button.className='car-thumbnail';
    button.title=label+' · увеличить';button.setAttribute('aria-label',label+' · '+(record.plate||'Номер не указан')+' · увеличить');
    const img=document.createElement('img');img.loading='lazy';img.decoding='async';
    img.width=80;img.height=50;img.alt=label;img.src=url+'&thumbnail=true';
    img.onerror=()=>{img.hidden=true;button.textContent='Фото недоступно';button.disabled=true;};
    button.append(img);
    button.onclick=e=>{
      e.stopPropagation();
      $('expandedMeasurement').src=url;
      $('measurementTitle').textContent=label;
      $('measurementCaption').textContent=(record.plate||'Номер не указан')+' · '+lengthText(record.length_m)+(field==='front_photo'?' · '+frontPhotoNote(record):'');
      $('measurementDialog').showModal();
    };
    td.append(button);
  }
  if(!td.children.length)td.textContent='Нет фото';
}
function tableRow(record,report=false){
  record=normalizeRecord(record);
  let tr=document.createElement('tr');
  if(!report){tr.classList.toggle('selected',current?.id===record.id);tr.onclick=()=>action(()=>selectRecord(record.id));}
  if(report)cell(tr,record.created_at.slice(0,10).split('-').reverse().join('.'));
  carPhotos(tr,record);
  cell(tr,record.created_at.slice(11,19));
  renderPlate(cell(tr,''),record);
  const lengthCell=cell(tr,recordLengthText(record)),note=measurementNote(record);
  if(note){const hint=document.createElement('small');hint.className='measurement-reason';hint.textContent=note;lengthCell.append(hint);}
  cell(tr,categories[record.category]);cell(tr,fmt(record.tariff.amount_rub)).className='tariff-only operator-price';cell(tr,'').append(badge(record.status));return tr;
}
function platePresentation(record){
  if(record.plate)return {text:record.plate};
  const o=record.plate_ocr,count=o?.candidate_count??o?.candidates?.length??0;
  if(o?.state==='queued')return {text:'Читаем номер…'};
  if(o?.state==='error')return {text:'Ошибка чтения номера',note:'Откройте запись и повторите распознавание.'};
  if(count>1)return {text:'Несколько вариантов',note:'Откройте запись и выберите номер по фото.'};
  if(count===1&&o.candidates?.[0]?.text)return {text:o.candidates[0].text,note:'Распознано · проверьте по фото'};
  if(o?.state==='not_found')return {text:'Номер не найден',note:'На сохранённых кадрах нет читаемого номера.'};
  return {text:record.front_photo?'Не указан':'Нет снимка номера'};
}
function renderPlate(target,record){
  const plate=platePresentation(record);target.replaceChildren();target.textContent=plate.text;
  if(plate.note){const hint=document.createElement('small');hint.className='plate-note';hint.textContent=plate.note;target.append(hint);}
}
async function loadVehicles(){
  const serial=++queueSerial,params=pageQuery();const result=await cachedGet('/vehicles?'+params);
  if(serial!==queueSerial||params.toString()!==pageQuery().toString())return;
  applyTariffMode(result.tariffs_enabled);
  if(queueOffset&&queueOffset>=result.count){queueOffset=Math.floor(Math.max(0,result.count-1)/250)*250;return loadVehicles();}
  if(queuePage!==result){
    rows=result.rows;$('carsTable').replaceChildren();rows.forEach(r=>$('carsTable').append(tableRow(r)));
    if(!rows.length)emptyTable('carsTable',7,'Автомобилей пока нет или они не подходят под фильтры');
    queuePage=result;pagination(result,'queue');
  }
  $('totalCars').textContent=result.count;$('filterCount').textContent=query().size||'';
  $('updatedAt').textContent='Обновлено: '+new Date().toLocaleTimeString('ru-RU');
  if(current){
    const id=current.id,detail=await cachedGet('/vehicles/'+id);
    if(current?.id!==id||serial!==queueSerial||detail.version<current.version)return;
    const stale=detail.version!==current.version;
    $('recordConflict').hidden=!stale;
    if(stale&&!dirty&&!referenceDirty)renderRecord(detail);
  }
}
function photo(id,emptyId,filename,version){
  $(id).hidden=!filename;$(emptyId).hidden=!!filename;
  if(filename){const src='/api/station/photos/'+encodeURIComponent(filename)+'?v='+version;if($(id).getAttribute('src')!==src)$(id).src=src;}
  else $(id).removeAttribute('src');
}
function showQuote(q){
  $('tariff').textContent=fmt(q.amount_rub)+(q.code?' · п. '+q.code:'');
  $('confirm').textContent=tariffsEnabled?'✓ Подтвердить':'✓ Подтвердить измерение';
  if(tariffsEnabled&&q.amount_rub!=null){const amount=document.createElement('span');amount.className='operator-price';amount.textContent=' • '+fmt(q.amount_rub);$('confirm').append(amount);}
  $('confirm').disabled=(tariffsEnabled&&q.amount_rub==null)||current?.status==='Оплачен';
  $('paid').disabled=!tariffsEnabled||current?.status!=='Подтвержден'||current?.tariff?.amount_rub==null;
  const notes=tariffsEnabled?[...(q.warnings||[])]:[];
  if(current?.source?.camera_note)notes.unshift(current.source.camera_note);
  for(const warning of current?.source?.measurement?.warnings||[])notes.unshift(warning);
  const measurementReason=current&&measurementNote(current);if(measurementReason)notes.unshift(measurementReason);
  if(current?.source)notes.unshift('Проверьте длину и категорию автомобиля.');
  $('warning').hidden=!notes.length;$('warningText').replaceChildren();
  notes.forEach(text=>{let small=document.createElement('small');small.textContent=text;$('warningText').append(small);});
  for(const category of (tariffsEnabled?q.category_options:[])||[]){
    const button=document.createElement('button');button.className='small-btn';button.type='button';button.textContent=categories[category];
    button.disabled=current?.status==='Оплачен';
    button.onclick=()=>{$('categoryInput').value=category;if(!$('reason').value.trim())$('reason').value='Уточнение типа автомобиля / состава';changed();};
    $('warningText').append(button);
  }
  $('boundaryTariffs').replaceChildren();
  tariffRows.filter(t=>t.category===$('categoryInput').value).forEach(t=>{let d=document.createElement('div');d.textContent=`${t.code}: ${t.description} — ${fmt(t.amount_rub)}`;$('boundaryTariffs').append(d);});
}
function normalizeRecord(r){return {...r,plate:russianPlate(r.plate),plate_ocr:r.plate_ocr?{...r.plate_ocr,candidates:r.plate_ocr.candidates?.map(p=>({...p,text:russianPlate(p.text)}))}:null};}
function renderRecord(r){
  applyTariffMode(r.tariffs_enabled);
  r=normalizeRecord(r);
  current=r;dirty=false;$('recordConflict').hidden=true;quoteSerial++;$('emptyDetail').hidden=true;$('recordDetail').hidden=false;
  renderOCR(r);
  renderCalibrationReference(r);
  renderPlate($('plate'),r);$('category').textContent=categories[r.category];
  $('detectedLength').textContent=r.source?recordLengthText({...r,length_m:r.measured_length_m}):'Ручная запись';
  $('recordStatus').replaceChildren(badge(r.status));$('plateInput').value=r.plate;$('categoryInput').value=r.category;
  $('lengthInput').value=r.length_m??'';$('capacityInput').value=r.load_capacity_t??'';capacityUI();$('manualTariff').value=r.manual_rub??'';$('reason').value='';
  mode=r.manual_rub!=null?'manual':'auto';modeUI();
  photo('selectedFront','selectedFrontEmpty',r.front_photo,r.version);$('expandFront').disabled=!r.front_photo;
  $('frontPhotoTiming').textContent=frontPhotoNote(r);
  $('expandSynchronizedFront').hidden=!r.source?.synchronized_front_photo;
  photo('selectedMeasurement','selectedMeasurementEmpty',r.side_photo,r.version);$('expandMeasurement').disabled=!r.side_photo;$('selectedPhotoFrame').textContent=r.source?`Кадр ${r.source.frame}`:'';
  photo('frontImage','frontEmpty',r.front_photo,r.version);photo('sideImage','sideEmpty',r.side_photo,r.version);
  $('measureLabel').hidden=r.measured_length_m==null;$('measureLabel').textContent='≈ '+lengthText(r.measured_length_m);
  $('sourceLabel').textContent=r.source?`Кадр ${r.source.frame}`:'Ручная запись';
  const closed=r.status==='Оплачен';
  ['plateInput','categoryInput','lengthInput','capacityInput','reason','autoMode','manualMode','reject','edit','confirm','uploadFront'].forEach(id=>$(id).disabled=closed);
  $('manualTariff').disabled=closed||mode!=='manual';$('paid').disabled=r.status!=='Подтвержден';
  showQuote(r.tariff);$('history').replaceChildren();
  if(tariffsEnabled&&r.tariff.mode==='disabled')recalc();
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
    picture.onclick=()=>{$('measurementTitle').textContent='Фото номера';$('expandedMeasurement').src=img.src;$('measurementCaption').textContent=p.text;$('measurementDialog').showModal();};
    const label=document.createElement('strong');label.textContent=p.text;
    const info=document.createElement('small');info.textContent=p.votes>1?'Проверен по нескольким снимкам':'Проверьте номер по фото';
    const use=document.createElement('button');use.className='small-btn';use.textContent='Использовать номер';use.disabled=['Оплачен','Подтвержден'].includes(r.status);
    use.onclick=()=>{$('plateInput').value=p.text;$('reason').value='Уточнение номера по фото';changed();};
    card.append(picture,label,info,use);$('ocrCandidates').append(card);
  }
}
$('recognizePlate').onclick=()=>action(async()=>{if(dirty)throw Error('Сначала сохраните изменения');renderRecord(await api('/vehicles/'+current.id+'/recognize-plate',{}));});
async function selectRecord(id){
  if(dirty||referenceDirty){toast('Сохраните изменения текущего автомобиля перед выбором другого.');return;}
  const record=await api('/vehicles/'+id);renderRecord(record);queuePage=null;await loadVehicles();
}
function fields(){
  const len=$('lengthInput').value.trim();const manual=$('manualTariff').value.trim();
  if(tariffsEnabled&&mode==='manual'&&!manual)throw Error('Укажите ручной тариф');
  return {plate:$('plateInput').value,category:$('categoryInput').value,length_m:len?Number(len):null,
    load_capacity_t:$('categoryInput').value==='truck_capacity'&&$('capacityInput').value.trim()?Number($('capacityInput').value):null,
    station_id:$('stationId').value,tariffs_enabled:tariffsEnabled,manual_rub:tariffsEnabled&&mode==='manual'?Number(manual):null,actor:$('actor').value.trim()||'Оператор',reason:$('reason').value.trim()};
}
async function recalc(){
  if(!current)return;const serial=++quoteSerial;$('confirm').disabled=true;
  try{const q=await api('/quote',fields());if(serial===quoteSerial){if(q.mode==='disabled')applyTariffMode(false);showQuote(q);}}catch(e){if(serial===quoteSerial){$('tariff').textContent=e.message;$('confirm').disabled=true;}}
}
function capacityUI(){$('capacityField').hidden=$('categoryInput').value!=='truck_capacity';}
function changed(){capacityUI();dirty=true;clearTimeout(quoteTimer);quoteSerial++;$('confirm').disabled=true;quoteTimer=setTimeout(recalc,150);}
function modeUI(){$('autoMode').classList.toggle('active',mode==='auto');$('manualMode').classList.toggle('active',mode==='manual');$('manualTariff').disabled=mode!=='manual';}
async function mutate(kind){
  if(!current)return;
  if(kind==='pay'&&dirty)throw Error('Перед отметкой оплаты сохраните и подтвердите изменения.');
  const r=await api('/vehicles/'+current.id,{...fields(),version:current.version,action:kind});
  renderRecord(await api('/vehicles/'+r.id));await loadVehicles();await syncClient();
  toast({edit:'Изменения сохранены. Запись требует подтверждения.',confirm:'Данные автомобиля подтверждены.',reject:'Автомобиль отклонён.',pay:'Полученная оплата отмечена.'}[kind]);
}
function clientQuery(){
  const params=new URLSearchParams({limit:clientPageSize,offset:clientOffset});
  if(clientOffset&&clientSnapshot!==null)params.set('snapshot',clientSnapshot);
  return params.toString();
}
function clientRow(record){
  const tr=document.createElement('tr');
  const plateCell=cell(tr,'');plateCell.className='client-number';
  const plate=document.createElement('strong');
  const reading=platePresentation(record);
  plate.textContent=russianPlate(record.plate||record.plate_ocr?.candidates?.[0]?.text);
  // Only show an OCR proposal when it is the sole, current candidate.
  if(!record.plate&&(record.plate_ocr?.state==='queued'||record.plate_ocr?.state==='error'||
      (record.plate_ocr?.candidate_count??record.plate_ocr?.candidates?.length??0)!==1))plate.textContent='';
  if(!plate.textContent){plate.textContent=reading.text==='Нет снимка номера'?'Номер не указан':reading.text;plate.className='client-placeholder';}
  plateCell.append(plate);
  const approved=record.status==='Подтвержден'||record.status==='Оплачен';
  const status=document.createElement('small');status.className='client-state '+(approved?'green':'orange');
  status.textContent=approved?'Подтвержден':'Не подтвержден';
  plateCell.append(status);
  const side=cell(tr,'');side.className='client-photo';
  if(record.side_photo){
    const img=document.createElement('img');img.alt='Фото автомобиля сбоку';img.decoding='async';
    img.src='/api/station/photos/'+encodeURIComponent(record.side_photo)+'?v='+record.version;
    img.onerror=()=>{side.replaceChildren();side.textContent='Фото недоступно';};side.append(img);
  }else side.textContent='Фото отсутствует';
  cell(tr,recordLengthText(record)).className='client-length'+(record.length_m==null?' client-placeholder':'');
  const price=cell(tr,'');price.className='client-price tariff-only';
  const amount=document.createElement('strong');
  const rejected=record.status==='Отклонён',value=record.tariff?.amount_rub;
  amount.textContent=rejected?'—':value==null?'Уточняется':fmt(value);
  if(value==null)amount.className='client-placeholder';
  price.append(amount);return tr;
}
async function syncClient(){
  const serial=++clientSerial,configPath=clientConfigurationPath();
  try{
    const config=await cachedGet(configPath);
    if(serial!==clientSerial||configPath!==clientConfigurationPath())return;
    applyClientConfiguration(config);
    const params=clientQuery();
    const data=await cachedGet('/vehicles?'+params);
    if(serial!==clientSerial||params!==clientQuery()||configPath!==clientConfigurationPath())return;
    applyTariffMode(data.tariffs_enabled);
    if(clientOffset&&clientOffset>=data.count){clientOffset=Math.floor(Math.max(0,data.count-1)/clientPageSize)*clientPageSize;return syncClient();}
    if(clientPage!==data){
      $('clientRows').replaceChildren();data.rows.forEach(r=>$('clientRows').append(clientRow(r)));
      clientPage=data;pagination(data,'client');
      $('clientEmpty').hidden=!!data.rows.length;$('clientData').hidden=!data.rows.length;
      $('clientLatest').disabled=clientOffset===0;
    }
    $('clientConnection').hidden=true;
    if(page==='client')$('systemStatus').textContent='● Список обновлён';
  }catch(e){
    if(serial===clientSerial){
      $('clientConnection').textContent=clientPage?'Нет связи с сервером. Показаны последние полученные данные.':'Нет связи с сервером. Ожидаем список автомобилей.';
      $('clientConnection').hidden=false;
    }
    throw e;
  }
}
for(const [id,direction] of [['clientPrev',-1],['clientNext',1]])$(id).onclick=()=>action(async()=>{
  clientOffset=Math.max(0,clientOffset+direction*clientPageSize);clientSnapshot=clientPage?.snapshot??null;await syncClient();
});
$('clientLatest').onclick=()=>action(async()=>{clientOffset=0;clientSnapshot=null;await syncClient();});
async function report(){
  const serial=++reportSerial,params=pageQuery(true);const data=await cachedGet('/vehicles?'+params);
  if(serial!==reportSerial||params.toString()!==pageQuery(true).toString())return;
  applyTariffMode(data.tariffs_enabled);
  if(reportOffset&&reportOffset>=data.count){reportOffset=Math.floor(Math.max(0,data.count-1)/250)*250;return report();}
  reportPage=data;pagination(data,'report');
  $('reportTable').replaceChildren();data.rows.forEach(r=>$('reportTable').append(tableRow(r,true)));
  if(!data.rows.length)emptyTable('reportTable',8,'За выбранный период записей нет');
  $('reportCount').textContent=data.count;$('reportAmount').textContent=fmt(data.amount_rub);
  $('reportPaid').textContent='Из них оплачено: '+fmt(data.paid_rub);$('reportIssues').textContent=data.issues;
}
function showPage(name){
  page=name;if(name!=='operator'&&ipMode){$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');}document.querySelectorAll('.page').forEach(el=>el.classList.toggle('active',el.id===name));
  document.querySelectorAll('[data-page]').forEach(el=>el.classList.toggle('active',el.dataset.page===name));
  if(name==='client')action(syncClient);if(name==='reports')action(report);
  if(name==='operator')action(loadVehicles);
  if(name==='calibration'&&!$('calibrationFrame').getAttribute('src'))$('calibrationFrame').src=$('calibrationFrame').dataset.src+'?target='+(ipMode?'ip':'operator');
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>showPage(b.dataset.page));
$('filterBtn').onclick=()=>$('filterPanel').classList.toggle('open');$('refresh').onclick=()=>action(async()=>{queueOffset=0;queueSnapshot=null;await loadVehicles();});
for(const [prefix,isReport] of [['queue',false],['report',true]])for(const [suffix,delta] of [['Prev',-250],['Next',250]])$(prefix+suffix).onclick=()=>action(async()=>{
 if(isReport){reportOffset=Math.max(0,reportOffset+delta);reportSnapshot=reportPage?.snapshot;await report();}
 else{queueOffset=Math.max(0,queueOffset+delta);queueSnapshot=queuePage?.snapshot;await loadVehicles();}
});
['filterPlate','filterStatus','filterCategory','filterMin','filterMax'].forEach(id=>$(id).oninput=()=>{clearTimeout(filterTimer);queueOffset=0;queueSnapshot=null;filterTimer=setTimeout(()=>action(loadVehicles),200);});
$('resetFilters').onclick=()=>{['filterPlate','filterStatus','filterCategory','filterMin','filterMax'].forEach(id=>$(id).value='');queueOffset=0;queueSnapshot=null;action(loadVehicles);};
['plateInput','categoryInput','lengthInput','capacityInput','manualTariff'].forEach(id=>$(id).oninput=changed);
$('reason').oninput=()=>{dirty=true;};
$('autoMode').onclick=()=>{mode='auto';modeUI();changed();};$('manualMode').onclick=()=>{mode='manual';modeUI();changed();};
['edit','confirm','reject'].forEach(kind=>$(kind).onclick=()=>action(()=>mutate(kind)));
$('paid').onclick=()=>action(()=>mutate('pay'));
$('historyToggle').onclick=()=>$('history').hidden=!$('history').hidden;
$('discardEdit').onclick=()=>action(async()=>{if(!current)return;renderRecord(await api('/vehicles/'+current.id));await loadVehicles();});
$('newRecord').onclick=()=>action(async()=>{
  if(dirty)throw Error('Сначала сохраните изменения текущего автомобиля.');
  const r=await api('/vehicles',{actor:$('actor').value.trim()||'Оператор'});renderRecord(r);queueOffset=0;queueSnapshot=null;await loadVehicles();
});
$('uploadFront').onclick=()=>$('frontFile').click();
$('frontFile').onchange=e=>action(async()=>{
  const file=e.target.files[0];if(!file||!current)return;
  if(dirty)throw Error('Сначала сохраните изменения текущего автомобиля.');
  const fd=new FormData();fd.append('file',file);fd.append('version',current.version);fd.append('actor',$('actor').value);
  const r=await api('/vehicles/'+current.id+'/front-photo',fd);renderRecord(await api('/vehicles/'+r.id));e.target.value='';toast('Фото сохранено');
});
['reportFrom','reportTo','reportStatus','reportCategory'].forEach(id=>$(id).onchange=()=>{reportOffset=0;reportSnapshot=null;action(report);});
$('resetReport').onclick=()=>{['reportFrom','reportTo','reportStatus','reportCategory'].forEach(id=>$(id).value='');reportOffset=0;reportSnapshot=null;action(report);};
$('exportCSV').onclick=()=>action(async()=>{
  await report();let a=document.createElement('a');a.href='/api/station/reports.csv?'+query(true);a.download='ferry_report.csv';a.click();
});
const requestedStation=new URLSearchParams(location.search).get('station');
$('stationId').value=requestedStation||localStorage.getItem('ferryStation')||('desk-'+Math.random().toString(36).slice(2,10));
function stationLink(){
  if(!/^[a-zA-Z0-9_-]{1,64}$/.test($('stationId').value)){toast('Место: латинские буквы, цифры, дефис или подчёркивание');return false;}
  localStorage.setItem('ferryStation',$('stationId').value);
  $('openClient').href='/?page=client&station='+encodeURIComponent($('stationId').value);return true;
}
if(!stationLink()){$('stationId').value='default';stationLink();}
$('stationId').onchange=()=>{if(stationLink()&&page==='client')action(syncClient);};
$('actor').value=localStorage.getItem('ferryOperator')||'Оператор';$('actor').onchange=()=>localStorage.setItem('ferryOperator',$('actor').value);
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===$('calibrationFrame').contentWindow&&e.data?.type==='station-capture'){action(loadVehicles);toast('Автомобиль добавлен в очередь оператора');}});
async function start(){
  const initial=new URLSearchParams(location.search).get('page');
  if(initial==='client'&&new URLSearchParams(location.search).get('settings')!=='1'){
    document.body.classList.add('client-only');showPage('client');
  }else{
    await syncConfiguration();
    const data=await api('/catalog');categories=data.categories;statuses=data.statuses;tariffRows=data.tariffs;
    options('categoryInput',categories);['filterCategory','reportCategory'].forEach(id=>options(id,categories,'Все категории'));
    ['filterStatus','reportStatus'].forEach(id=>options(id,Object.fromEntries(statuses.map(s=>[s,s])),'Все статусы'));
    $('tariffPolicy').textContent=data.policy;tariffRows.forEach(t=>{let tr=document.createElement('tr');[t.code,t.category_label,t.description,fmt(t.amount_rub)].forEach(v=>cell(tr,v));$('tariffTable').append(tr);});
    const health=await api('/health');$('systemStatus').textContent=health.gpu_available?'● Система готова':'⚠ Система недоступна';$('systemStatus').title=health.gpu??'';
    if(initial==='client')showPage('client');else await loadVehicles();
  }
  setInterval(async()=>{if(pending||document.hidden||pollBusy)return;pollBusy=true;try{if(page==='client')await syncClient();else if(page==='operator')await loadVehicles();else await syncConfiguration();}catch(e){$('systemStatus').textContent='⚠ Нет связи с сервером';}finally{pollBusy=false;}},3000);
}
let ipMode=false,ipRunning=false,ipPollTimer=null;
let detectorEditDirty=false,detectorApplying=false;
let liveSource=null, liveGeneration=0, liveSnapshot=null, importedProfile=null;
let liveTracks=new LiveTracks(), streamId=null, streamFrame=0, streamTimer=null, streamStarting=false,streamResultCursor=0;
function showDetector(detector,context){
 const names={'yolo26n':'YOLO26 N','yolo26m':'YOLO26 M','yolo26l':'YOLO26 L','rtdetr-l':'RT-DETR L','rtdetr-x':'RT-DETR X'};
 $('activeDetector').textContent=detector?.model?`${names[detector.model]||detector.model} · ${detector.imgsz} px`:'Не определена';
 $('activeDetectorContext').textContent=context;
 if(detector?.model&&!detectorEditDirty&&!detectorApplying){$('operatorDetectorModel').value=detector.model;$('operatorDetectorSize').value=detector.imgsz;}
}
function selectedVideoDetector(){
 if(ipMode||streamId)return;
 const p=liveSource?.profile||importedProfile;
 showDetector(p?{model:p.detector_model??'yolo26n',imgsz:p.detector_imgsz??640}:null,'Видеозаписи · выбранная настройка');
}
function calibrationLabel(){const p=liveSource?.profile||importedProfile;selectedVideoDetector();if(p?.wheel_recovery_calibration){$('calibrationStatus').textContent=`Контур и колёса · резерв для разных положений · приблизительная длина`;return;}if(p?.wheel_calibration){$('calibrationStatus').textContent=`Контур и колёса · ${p.wheel_calibration.training_count} каталожных проездов · приблизительная длина`;return;}if(p?.outline_calibration){$('calibrationStatus').textContent=`Контур кузова · ${p.outline_calibration.training_count} каталожных проездов · приблизительная длина`;return;}if(p?.survey_calibration){$('calibrationStatus').textContent='Калибровка по метровым отметкам · требуется проверка геометрии';return;}$('calibrationStatus').textContent=(p?.references?.length||p?.metric_rulers?.length)?`Калибровка активна · ${p.metric_rulers?.length? p.metric_rulers.length+' мерных линий':p.references.length+' эталонов'} · измерение у линии`:'Импортируйте JSON с дорогой и эталонными длинами';}
async function workbench(path,body){const response=await fetch('/api/workbench'+path,body instanceof FormData?{method:'POST',body}:body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
async function stopStream(){const id=streamId;streamId=null;liveGeneration++;clearTimeout(streamTimer);$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('streamPlay').textContent='▶ Пуск';selectedVideoDetector();if(id)await fetch('/api/stream/'+id,{method:'DELETE'});}
async function startStream({throwOnError=false}={}){
 if(streamStarting||streamId)return;if(!liveSource){toast('Откройте видео');return;}streamStarting=true;
 if(liveSource.profile.measurement_line_x==null)liveSource.profile.measurement_line_x=liveSource.profile.image_size[0]/2;
 try{const response=await fetch('/api/stream/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({media_id:liveSource.media.id,profile:liveSource.profile,frame:streamFrame,front_media_id:liveSource.frontMedia?.id,front_offset_seconds:liveSource.frontOffset??3})});const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));streamId=data.id;streamResultCursor=0;liveGeneration++;liveTracks=new LiveTracks();liveSnapshot=null;$('operatorDetection').src='/api/stream/'+streamId+'/video';if(liveSource.frontMedia)$('frontStream').src='/api/stream/'+streamId+'/front';$('streamPlay').textContent='Ⅱ Пауза';pollStream(streamId,liveGeneration);}
 catch(e){toast(e.message);if(throwOnError)throw e;}finally{streamStarting=false;}
}
async function pollStream(id,generation){
 if(id!==streamId||generation!==liveGeneration)return;
 try{const response=await fetch('/api/stream/'+id+'/state?after_sequence='+streamResultCursor);if(!response.ok)throw Error('Поток завершён. Нажмите Пуск для повторного запуска.');const state=await response.json();if(id!==streamId||generation!==liveGeneration)return;
 if(!ipMode)showDetector(state.detector,state.ended?'Видеозапись завершена':'Видеозаписи · настройка сервера');
 streamFrame=state.frame;$('frontStatus').textContent=state.front_seconds==null?'Не подключена':state.plate_error?'Номер временно недоступен':(state.plates?.candidates?.map(p=>p.text).join(', ')||'Камера работает');$('syncStatus').textContent=state.sync_error_ms==null?'Одна камера':'Камеры синхронизированы';$('streamSeek').value=100*streamFrame/(liveSource.media.frames-1);
 $('detectionStatus').textContent=state.error?'Не удалось обработать изображение':'Измерение работает';
 const source=structuredClone(liveSource),results=state.results??(state.result?[state.result]:[]);
 if(state.result_gap)toast('Обработка кадров отстала. Проверьте последние проезды.');
 for(const result of results){if(result.frame===liveSnapshot?.frame)continue;const frame=result.frame;
 streamResultCursor=Math.max(streamResultCursor,result.sequence??0);
 liveSnapshot={source,frame,detections:result.detections};
 $('liveCar').replaceChildren();result.detections.forEach((d,i)=>{const option=document.createElement('option');option.value=i;option.textContent=(i+1)+': '+d.label+' · '+(d.length_m==null?'длина не измерена':d.length_m.toFixed(2)+' м');$('liveCar').append(option);});$('captureLive').disabled=!result.detections.some(captureCandidate);
      const tracks=liveTracks.update(result.detections,frame/source.media.fps);
      if($('autoMeasure').checked){for(let i=0;i<result.detections.length;i++){const d=result.detections[i],t=tracks[i];if(t.sent||d.passage_captured||d.capture_ready===false)continue;
        if(captureCandidate(d)){t.sent=true;queueLive({source,frame},d).catch(e=>{t.sent=false;toast(e.message);});continue;}
        const line=source.profile.measurement_line_x,prev=t.previous;
        if(line!=null&&prev&&d.depth!=null){const before=prev.box[0]+prev.box[2]/2-line,after=d.bbox[0]+d.bbox[2]/2-line;
          if((before*after<0||Math.abs(before)<=(source.profile.line_tolerance_px??10))&&frame/source.media.fps-prev.time<2){t.sent=true;
          captureCrossing(source,frame,d,prev).then(ok=>{if(!ok)t.sent=false;}).catch(e=>{t.sent=false;toast(e.message);});}}
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
$('captureLive').onclick=()=>action(async()=>{const snapshot=structuredClone(liveSnapshot),d=snapshot?.detections[Number($('liveCar').value)];if(d&&captureCandidate(d))await queueLive(snapshot,d);});
try{const saved=JSON.parse(localStorage.getItem('ferryVideo'));if(saved?.media&&saved?.profile){liveSource=saved;calibrationLabel();$('liveStatus').textContent=saved.media.name;}}catch{}
window.addEventListener('pagehide',()=>{if(streamId)fetch('/api/stream/'+streamId,{method:'DELETE',keepalive:true});});
function captureCandidate(d){return d.at_measurement_line===true&&(d.length_m!=null||['outside_calibration','calibration_review'].includes(d.status));}
async function queueLive(snapshot,d){
  const record=await api('/capture',{media_id:snapshot.source.media.id,profile:snapshot.source.profile,frame:snapshot.frame,bbox:d.bbox,label:d.label,source:snapshot.source.profile.detector_model??'yolo26n',temporal:true,passage_id:d.passage_id,actor:$('actor').value||'Оператор',front_media_id:snapshot.source.frontMedia?.id,front_session_id:streamId,capture_session_id:streamId,front_offset_seconds:snapshot.source.frontOffset??3});
  await loadVehicles();if(!dirty&&!current)renderRecord(await api('/vehicles/'+record.id));return record;
}
$('operatorCalibration').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{
  const parsed=JSON.parse(await file.text());const profile=await workbench('/validate-profile',parsed.profile||parsed);
  if(!ipMode&&liveSource&&profile.image_size.toString()!==liveSource.media.image_size.toString())throw Error('Разрешение калибровки не совпадает с видео');
  if(ipMode){const result=await ipApi('/calibration',profile);detectorEditDirty=false;await ipPoll();$('calibrationStatus').textContent='Настройка камер сохранена';toast(result.restarted?'Настройка применена. Камеры переподключены.':'Настройка сохранена для подключения камер.');notifyCalibration('ip');e.target.value='';return;}
  if(streamId)await workbench('/detector/check',detectorChoice(profile));
  importedProfile=await workbench('/operator-calibration',profile);detectorEditDirty=false;
  if(!ipMode&&liveSource){const running=!!streamId;await stopStream();liveSource.profile=importedProfile;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));if(running)await startStream({throwOnError:true});}
  notifyCalibration('operator');calibrationLabel();toast('Калибровка загружена. Измерение включено.');$('autoMeasure').checked=true;
}catch(error){toast(error.message);}e.target.value='';};


async function captureCrossing(source,frame,original,previous){
  const result=await api('/capture-crossing',{media_id:source.media.id,profile:source.profile,
    frame,bbox:original.bbox,label:original.label,passage_id:original.passage_id,source:source.profile.detector_model??'yolo26n',temporal:true,
    before_frame:Math.round(previous.time*source.media.fps),before_bbox:previous.box,
    actor:$('actor').value||'Оператор',front_media_id:source.frontMedia?.id,front_session_id:streamId,capture_session_id:streamId,
    front_offset_seconds:source.frontOffset??3});
  if(!result.captured){toast(result.reason||'Не удалось восстановить проезд у линии. Проверьте вручную.');return false;}
  await loadVehicles();if(!dirty&&!current)renderRecord(await api('/vehicles/'+result.record.id));return true;
}

$('expandMeasurement').onclick=()=>{if(!current?.side_photo)return;$('measurementTitle').textContent='Фото измеренного автомобиля';$('expandedMeasurement').src=current.full_frame_photo?'/api/station/photos/'+encodeURIComponent(current.full_frame_photo):$('selectedMeasurement').src;$('measurementCaption').textContent=`${current.plate||'Номер не указан'} · ${lengthText(current.measured_length_m)}${current.source?' · кадр '+current.source.frame:''}`;$('measurementDialog').showModal();};
$('closeMeasurement').onclick=()=>$('measurementDialog').close();
$('measurementDialog').onclick=e=>{if(e.target!==$('measurementDialog'))return;const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();};

$('expandFront').onclick=()=>{if(!current?.front_photo)return;$('measurementTitle').textContent='Фото спереди';$('expandedMeasurement').src=$('selectedFront').src;$('measurementCaption').textContent=frontPhotoNote(current);$('measurementDialog').showModal();};
$('expandSynchronizedFront').onclick=()=>{const filename=current?.source?.synchronized_front_photo;if(!filename)return;$('measurementTitle').textContent='Синхронное фото спереди';$('expandedMeasurement').src='/api/station/photos/'+encodeURIComponent(filename)+'?v='+current.version;$('measurementCaption').textContent='Фронтальная камера в момент бокового измерения. Сравните с выбранным фото номера.';$('measurementDialog').showModal();};
$('frontOffset').oninput=()=>{$('frontOffsetValue').textContent=(Number($('frontOffset').value)>=0?'+':'')+Number($('frontOffset').value).toFixed(2)+' с';};
$('frontOffset').onchange=async()=>{if(!liveSource)return;const running=!!streamId;await stopStream();liveSource.frontOffset=Number($('frontOffset').value);localStorage.setItem('ferryVideo',JSON.stringify(liveSource));if(running)await startStream();};
$('frontVideoFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const fd=new FormData();fd.append('file',file);const frontMedia=await workbench('/media',fd);if(frontMedia.frames<2)throw Error('Выберите видео');const running=!!streamId;await stopStream();if(!liveSource)throw Error('Сначала выберите боковое видео');liveSource.frontMedia=frontMedia;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));$('frontStatus').textContent=frontMedia.name;if(running)await startStream();}catch(e){toast(e.message);}e.target.value='';};
$('streamPlay').disabled=true;
async function restoreVideoSettings(){
 const [pairResult,profileResult]=await Promise.allSettled([
  fetch('/api/stream/configuration').then(async response=>response.ok?response.json():null),
  workbench('/operator-calibration')
 ]);
 // A late startup response must never overwrite a selection or active stream.
 if(streamId||streamStarting||detectorApplying||detectorEditDirty||importedProfile)return;
 const pair=pairResult.status==='fulfilled'?pairResult.value:null;
 const stored=profileResult.status==='fulfilled'?profileResult.value:pair?.profile;
 if(!liveSource&&pair)liveSource=pair;
 if(stored)importedProfile=stored;
 if(liveSource){
  if(stored&&(!liveSource.media.image_size||liveSource.media.image_size.toString()===stored.image_size.toString()))liveSource.profile=stored;
  localStorage.setItem('ferryVideo',JSON.stringify(liveSource));
  if(!ipMode){$('liveStatus').textContent=liveSource.media.name;$('frontStatus').textContent=liveSource.frontMedia?.name||'Не подключена';}
  $('frontOffset').value=liveSource.frontOffset??3;$('frontOffset').oninput();
 }
 if(!ipMode)calibrationLabel();
}
restoreVideoSettings().catch(e=>toast(e.message)).finally(()=>{if(!detectorApplying)$('streamPlay').disabled=false;});

$('plateInput').addEventListener('input',()=>{const field=$('plateInput'),pos=field.selectionStart;field.value=russianPlate(field.value);field.setSelectionRange(pos,pos);});

async function ipApi(path,body){
 const r=await fetch('/api/ip'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Проверьте настройки камер');return data;
}
async function ipPoll(){
 clearTimeout(ipPollTimer);if(!ipMode)return;
 if(document.hidden||page!=='operator'){ipPollTimer=setTimeout(ipPoll,1500);return;}
 try{const s=await ipApi('/state?compact=true');if(!ipMode)return;const wasRunning=ipRunning;ipRunning=s.running;
 showDetector(s.detector,s.running?'IP-камеры · настройка сервера':'IP-камеры · камеры остановлены');
 $('streamPlay').textContent=s.running?'■ Остановить камеры':'▶ Подключить камеры';
 $('ipStatus').textContent=!s.running?'Камеры остановлены':s.error||(!s.side_ready&&!s.front_ready?'Нет связи с камерами':!s.front_ready?'Боковая камера работает · фронтальная недоступна':!s.side_ready?'Фронтальная камера работает · боковая недоступна':'Камеры работают');
 if(s.running){
  if(!wasRunning||ipSessionId!==s.id||!$('operatorDetection').getAttribute('src')){$('operatorDetection').src='/api/ip/side/video?session='+s.id;$('frontStream').src='/api/ip/front/video?session='+s.id;ipSessionId=s.id;}
  $('operatorDetection').hidden=!s.side_ready;$('frontStream').hidden=!s.front_ready;
  $('liveStatus').textContent=s.side;$('frontStatus').textContent=s.front_ready?(s.plate_error?'Ошибка чтения номера':s.plates?.candidates?.map(p=>p.text).join(', ')||s.front):s.front;
  $('syncStatus').textContent=s.ready?'Изображения согласованы':s.side_ready?'Длина измеряется без фронтального снимка':s.front_ready?'Номера читаются · длина недоступна':'Ожидание камер';
  $('detectionStatus').textContent=s.error||(!s.side_ready?'Ожидание боковой камеры':s.auto_measure?'Автоматическое измерение включено':'Автоматическое измерение выключено');
  $('autoMeasure').checked=s.auto_measure;
 }else{$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('liveStatus').textContent='Не подключена';$('frontStatus').textContent='Не подключена';$('syncStatus').textContent='Ожидание подключения';$('detectionStatus').textContent='Камеры остановлены';}
 }catch(e){$('ipStatus').textContent='Нет связи с программой';if(ipMode)showDetector(null,'Нет связи с сервером');}
 if(ipMode)ipPollTimer=setTimeout(ipPoll,500);
}
async function toggleIp(){
 $('streamPlay').disabled=true;
 try{await ipApi(ipRunning?'/stop':'/start',{});await ipPoll();}catch(e){toast(e.message);}finally{$('streamPlay').disabled=false;}
}
$('sourceMode').onchange=()=>action(async()=>{
 const next=$('sourceMode').value==='ip';
 detectorEditDirty=false;$('operatorDetectorStatus').textContent='Выберите модель и нажмите «Применить модель». Разметка дороги сохранится.';
 if(next){await stopStream();ipMode=true;}else{ipRunning=false;ipMode=false;clearTimeout(ipPollTimer);$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');$('streamPlay').textContent='▶ Пуск';}
 document.body.classList.toggle('ip-mode',ipMode);$('ipSettings').hidden=!ipMode;
 if(ipMode)showDetector(null,'IP-камеры · загрузка настройки');
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
if(new URLSearchParams(location.search).get('page')!=='client')ipApi('/configuration').then(async cfg=>{
 $('ipOffset').value=cfg.offset_seconds;$('ipTolerance').value=cfg.tolerance_ms;$('ipAutostart').checked=cfg.autostart;
 $('ipSideUrl').placeholder=cfg.side_configured?'Адрес сохранён':'rtsp://…';$('ipFrontUrl').placeholder=cfg.front_configured?'Адрес сохранён':'rtsp://…';
 const state=await ipApi('/state');if(state.running||localStorage.getItem('ferrySourceMode')==='ip'){$('sourceMode').value='ip';$('sourceMode').onchange();}
}).catch(e=>toast(e.message));

document.addEventListener('visibilitychange',()=>{if(document.hidden&&ipMode){$('operatorDetection').removeAttribute('src');$('frontStream').removeAttribute('src');}else if(ipMode)ipPoll();});


window.addEventListener('message',e=>{
  if(e.origin!==location.origin||e.source!==$('calibrationFrame').contentWindow||e.data?.type!=='station-calibration-saved')return;
  action(async()=>{
    if(e.data.target==='ip'){$('calibrationStatus').textContent='Калибровка IP-камер применена.';if(ipMode)await ipPoll();return;}
    importedProfile=e.data.profile;
    if(liveSource&&liveSource.media.image_size.toString()===importedProfile.image_size.toString()){
      const running=!!streamId;if(running){await workbench('/detector/check',detectorChoice(importedProfile));await stopStream();}
      liveSource.profile=importedProfile;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));
      if(running)await startStream({throwOnError:true});
    }
    calibrationLabel();toast('Калибровка сохранена для видеозаписей');
  });
});

function renderCalibrationReference(r){
  referenceDirty=false;
  const approved=['Подтвержден','Оплачен'].includes(r.status),ref=r.calibration_reference?.reference;
  $('referenceLength').value=ref?.length_m??'';
  $('referenceVerified').checked=false;
  $('referenceLength').disabled=!approved;$('referenceVerified').disabled=!approved;
  $('verifyReference').disabled=!approved||!r.source?.bbox||!r.side_photo;
  $('removeReference').hidden=!ref;
  $('verifyReference').textContent=ref?'Обновить проверенный эталон':'Добавить проверенный эталон';
  $('referenceStatus').textContent=ref?`Эталон: ${ref.length_m} м · проверил ${ref.verified_by}`:approved?'Введите независимо проверенную длину. Оценка камеры не подставляется.':'Сначала подтвердите автомобиль. Неподтверждённые записи не участвуют в калибровке.';
}
async function saveCalibrationReference(enabled){
  if(!current||dirty)throw Error('Сначала сохраните изменения и подтвердите автомобиль.');
  if(enabled&&(!$('referenceVerified').checked||!Number($('referenceLength').value)))throw Error('Введите фактическую длину и подтвердите проверку.');
  renderRecord(await api('/vehicles/'+current.id+'/calibration-reference',{
    version:current.version,actor:$('actor').value.trim()||'Оператор',enabled,
    actual_length_m:enabled?Number($('referenceLength').value):null,verified:$('referenceVerified').checked
  }));
  toast(enabled?'Проверенный эталон сохранён. Загрузите или скачайте совместимую калибровку.':'Эталон исключён. Скачайте и примените обновлённую калибровку.');
  await loadVehicles();
}
$('verifyReference').onclick=()=>action(()=>saveCalibrationReference(true));
$('removeReference').onclick=()=>action(()=>saveCalibrationReference(false));

$('referenceLength').oninput=()=>{referenceDirty=true;};
$('referenceVerified').onchange=()=>{referenceDirty=true;};

function detectorChoice(profile){return {detector_model:profile.detector_model??'yolo26n',detector_imgsz:profile.detector_imgsz??640};}
for(const id of ['operatorDetectorModel','operatorDetectorSize'])$(id).onchange=()=>{
 detectorEditDirty=true;$('operatorDetectorStatus').textContent='Выбор ещё не применён. Нажмите «Применить модель».';
};
function notifyCalibration(target){$('calibrationFrame').contentWindow?.postMessage({type:'station-settings-changed',target},location.origin);}
$('applyDetector').onclick=async()=>{
 if(detectorApplying)return;
 detectorApplying=true;
 const target=ipMode?'ip':'operator';
 const choice={detector_model:$('operatorDetectorModel').value,detector_imgsz:Number($('operatorDetectorSize').value)};
 const controls=['applyDetector','operatorDetectorModel','operatorDetectorSize','sourceMode','operatorCalibration','streamPlay'];
 const disabled=controls.map(id=>$(id).disabled);
 controls.forEach(id=>$(id).disabled=true);
 $('operatorDetectorStatus').textContent='Проверяем и применяем модель… Первая загрузка весов может занять несколько минут.';
 try{
  if(target==='ip'){
   const result=await ipApi('/detector',choice);
   showDetector(result.detector,result.running?'IP-камеры · настройка сервера':'IP-камеры · камеры остановлены');
   $('operatorDetectorStatus').textContent=result.restarted?'Модель применена. Камеры переподключены.':'Модель сохранена. Будет использоваться при подключении камер.';
   await ipPoll();
  }else{
   const previous=liveSource?.profile||importedProfile||await workbench('/operator-calibration');
   const next={...structuredClone(previous),...choice};
   const running=!!streamId;
   if(running)await workbench('/detector/check',choice);
   importedProfile=await workbench('/operator-calibration',next);
   if(running)await stopStream();
   if(liveSource){liveSource.profile=importedProfile;localStorage.setItem('ferryVideo',JSON.stringify(liveSource));}
   if(running)await startStream({throwOnError:true});
   calibrationLabel();
   $('operatorDetectorStatus').textContent=running?'Модель применена. Видео продолжает работать.':'Модель сохранена для видеозаписей.';
  }
  detectorEditDirty=false;notifyCalibration(target);
 }catch(e){$('operatorDetectorStatus').textContent='Не удалось применить: '+e.message;toast(e.message);}
 finally{detectorApplying=false;controls.forEach((id,i)=>$(id).disabled=disabled[i]);}
};
// Start after all page and camera state has been initialized.
start().catch(e=>{toast(e.message);$('systemStatus').textContent='⚠ Ошибка подключения';});
