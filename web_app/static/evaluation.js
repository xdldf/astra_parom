'use strict';
const $=id=>document.getElementById(id);
let current=null,offset=0,total=0,busy=false;
const photoUrl=name=>'/api/station/photos/'+encodeURIComponent(name);
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function lock(value){busy=value;for(const id of ['save','skip','refresh','state','previous','next'])$(id).disabled=value; if(!value){$('previous').disabled=offset===0;$('next').disabled=offset+1>=total;}}
async function api(path,body){
  const response=await fetch('/api/station/evaluation'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  const data=await response.json();
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'Не удалось сохранить. Проверьте длину и обновите кадр.');
  return data;
}
function render(record){
  current=record;$('review').hidden=!record;$('empty').hidden=!!record;
  $('count').textContent=total?`${offset+1} из ${total}`:'0 кадров';
  if(!record)return;
  $('identity').textContent=`${record.plate||'Номер не указан'} · ${record.created_at.replace('T',' ').slice(0,19)}`;
  $('photo').src=photoUrl(record.full_frame_photo);$('full').href=$('photo').src;
  $('export').hidden=false;$('export').href='/api/station/evaluation/'+encodeURIComponent(record.id)+'/calibration.json';
  const [x,y,w,h]=record.bbox,[width,height]=record.image_size;
  $('photo').parentElement.style.maxWidth=`calc(56vh * ${width/height})`;
  Object.assign($('box').style,{left:100*x/width+'%',top:100*y/height+'%',width:100*w/width+'%',height:100*h/height+'%'});
  const outline=record.measurement?.outline;
  $('outline').setAttribute('viewBox',`0 0 ${width} ${height}`);
  $('outlinePoints').setAttribute('points',(outline?.contour||[]).map(p=>p.join(',')).join(' '));
  $('outline').hidden=!$('showBox').checked||!outline?.contour?.length;
  $('basis').textContent=outline?(outline.status==='estimated'?'По контуру кузова. ':outline.status==='not_applicable'?'Оценка рамки; контурная модель предназначена для легковых автомобилей. ':'Контурная оценка не назначена. ')+
    (outline.baseline_length_m!=null?`Исходная оценка рамки: ${outline.baseline_length_m.toFixed(3)} м.`:''):'';
  $('measured').textContent=record.measured_length_m==null?'Не измерена':record.measured_length_m.toFixed(3)+' м';
  $('quality').textContent=record.measurement?.warnings?.join(' ')||'';
  $('actual').value=record.evaluation?.actual_length_m??'';
  $('datasetRole').value=record.evaluation?.dataset_role||'calibration';
  $('datasetRole').disabled=record.evaluation?.dataset_assigned??record.evaluation?.state==='labeled';
  $('physicalId').value=record.evaluation?.physical_vehicle_id||'';
  $('physicalId').disabled=!!$('physicalId').value&&$('datasetRole').disabled;
  $('referenceSource').value=record.evaluation?.reference_source||'unknown';
  $('uncertainty').value=record.evaluation?.reference_uncertainty_m??'';
  referenceRequirements();
}
function referenceRequirements(){
  const validation=$('datasetRole').value==='validation';
  $('physicalId').required=validation;$('uncertainty').required=validation;
}
async function load(){
  lock(true);
  try{
    let data=await api('?state='+$('state').value+'&offset='+offset+'&limit=1');
    if(offset&&offset>=data.count){offset=Math.max(0,data.count-1);data=await api('?state='+$('state').value+'&offset='+offset+'&limit=1');}
    total=data.count;render(data.rows[0]||null);
  }catch(e){message(e.message,true);}finally{lock(false);}
}
async function submit(action){
  if(busy||!current)return;
  if(action==='label'&&!$('form').reportValidity())return;
  const actor=$('actor').value.trim();if(!actor){message('Введите имя оператора.',true);return;}
  if(action==='label'&&$('datasetRole').value==='validation'&&$('referenceSource').value!=='physical_measurement'){
    message('Для независимой проверки нужен физический замер.',true);return;
  }
  lock(true);
  try{
    const record=await api('/'+encodeURIComponent(current.id),{version:current.version,actor,action,
      actual_length_m:action==='label'?Number($('actual').value):null,
      physical_vehicle_id:$('physicalId').value.trim()||null,dataset_role:$('datasetRole').value,
      reference_source:$('referenceSource').value,
      reference_uncertainty_m:$('uncertainty').value===''?null:Number($('uncertainty').value)});
    localStorage.setItem('evaluationActor',actor);
    message(action==='skip'?'Кадр пропущен. Его можно открыть в очереди «Пропущенные».':
      record.evaluation.dataset_role==='validation'?'Длина сохранена для независимой проверки. Она не используется в калибровке.':
      record.evaluation.reference_added?'Фактическая длина сохранена в данных оценки и эталонах JSON.':
      'Фактическая длина сохранена в данных оценки JSON. Эта рамка не добавлена в эталоны масштаба.');
    if($('state').value===record.evaluation.state)offset++;
    await load();
  }catch(e){message(e.message,true);}finally{lock(false);}
}
$('form').onsubmit=e=>{e.preventDefault();submit('label');};
$('skip').onclick=()=>submit('skip');
$('refresh').onclick=()=>{message('');load();};
$('state').onchange=()=>{offset=0;message('');load();};
$('previous').onclick=()=>{offset=Math.max(0,offset-1);load();};
$('next').onclick=()=>{offset++;load();};
$('showBox').onchange=()=>{$('box').hidden=!$('showBox').checked;$('outline').hidden=!$('showBox').checked||!current?.measurement?.outline?.contour?.length;};
$('datasetRole').onchange=referenceRequirements;
$('actor').value=localStorage.getItem('evaluationActor')||localStorage.getItem('ferryActor')||'Оператор';
load();
