// Short-lived overlap association: one automatic capture per visible passage.
class LiveTracks {
  constructor(){this.rows=[];this.next=1;}
  update(detections,time){
    this.rows=this.rows.filter(t=>time-t.time<3);
    const used=new Set();
    return detections.map(d=>{
      let best=null,score=.2;
      for(const t of this.rows){
        if(used.has(t.id))continue;
        const [x,y,w,h]=d.bbox,[a,b,c,e]=t.box;
        const area=Math.max(0,Math.min(x+w,a+c)-Math.max(x,a))*Math.max(0,Math.min(y+h,b+e)-Math.max(y,b));
        const iou=area/(w*h+c*e-area);
        if(iou>score){score=iou;best=t;}
      }
      if(!best){best={id:this.next++,sent:false};this.rows.push(best);}
      best.previous=best.box?{box:best.box,time:best.time}:null;best.box=d.bbox;best.time=time;used.add(best.id);return best;
    });
  }
}
if(typeof module!=='undefined')module.exports=LiveTracks;
