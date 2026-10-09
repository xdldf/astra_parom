// Short-lived overlap association: one automatic capture per visible passage.
class LiveTracks {
  constructor(){this.rows=[];this.next=1;}
  update(detections,time){
    this.rows=this.rows.filter(t=>t.missingSince==null||time-t.missingSince<2);
    const used=new Set();
    const assigned=detections.map(d=>{
      let best=d.passage_id?this.rows.find(t=>t.passageId===d.passage_id):null,score=.2;
      for(const t of this.rows){
        if(d.passage_id)break;
        if(used.has(t.id))continue;
        const [x,y,w,h]=d.bbox,[a,b,c,e]=t.box;
        const area=Math.max(0,Math.min(x+w,a+c)-Math.max(x,a))*Math.max(0,Math.min(y+h,b+e)-Math.max(y,b));
        const iou=area/(w*h+c*e-area);
        if(iou>score){score=iou;best=t;}
      }
      if(!best&&!d.passage_id){
        const possible=this.rows.filter(t=>!used.has(t.id)&&time>t.time&&time-t.time<=2&&LiveTracks.plausible(t.box,d.bbox));
        if(possible.length===1&&detections.filter(other=>LiveTracks.plausible(possible[0].box,other.bbox)).length===1)best=possible[0];
      }
      if(!best){best={id:this.next++,passageId:d.passage_id,sent:false};this.rows.push(best);}
      best.previous=best.box?{box:best.box,time:best.time}:null;best.box=d.bbox;best.time=time;best.missingSince=null;
      if(d.passage_captured)best.sent=true;
      used.add(best.id);return best;
    });
    for(const t of this.rows)if(!used.has(t.id)&&t.missingSince==null)t.missingSince=time;
    return assigned;
  }
  static plausible(a,b){
    const [x,y,w,h]=a,[u,v,c,d]=b;
    return Math.min(w,c)>0&&Math.min(h,d)>0&&w/c>=.6&&w/c<=1.67&&h/d>=.6&&h/d<=1.67&&
      Math.min(y+h,v+d)-Math.max(y,v)>=.6*Math.min(h,d)&&Math.abs(x+w/2-u-c/2)<=2*Math.max(w,c);
  }
}
if(typeof module!=='undefined')module.exports=LiveTracks;
