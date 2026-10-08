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
  $('measured').textContent=record.measured_length_m==null?'Не измерена':record.measured_length_m.toFixed(3)+' м';
  $('quality').textContent=record.measurement?.warnings?.join(' ')||'';
  $('actual').value=record.evaluation?.actual_length_m??'';
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
  lock(true);
  try{
    const record=await api('/'+encodeURIComponent(current.id),{version:current.version,actor,action,
      actual_length_m:action==='label'?Number($('actual').value):null});
    localStorage.setItem('evaluationActor',actor);
    message(action==='skip'?'Кадр пропущен. Его можно открыть в очереди «Пропущенные».':
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
$('showBox').onchange=()=>{$('box').hidden=!$('showBox').checked;};
$('actor').value=localStorage.getItem('evaluationActor')||localStorage.getItem('ferryActor')||'Оператор';
load();
