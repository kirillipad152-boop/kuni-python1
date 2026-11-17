"""Standalone HTML viewer of the memory graph (replaces the `--debug` diary window of the C++ kernel).

One self-contained file, no CDN: a force-directed layout on a canvas, search, click a node to read its description,
facts and diary pages.
"""

from __future__ import annotations

import json

from .memory.graph import MENTIONS, GraphMemory

MAX_NODES = 500


def graph_payload(memory: GraphMemory) -> dict:
    nodes = sorted(memory.nodes.values(), key=lambda n: len(memory._adj.get(n.id, ())), reverse=True)  # noqa: SLF001
    keep = {n.id for n in nodes[:MAX_NODES]}
    out_nodes = []
    for n in nodes[:MAX_NODES]:
        facts = []
        pages = []
        for e, other in memory.neighbors(n.id):
            if e.relation == MENTIONS:
                if other.is_episode:
                    pages.append(other.summary)
                elif n.is_episode:
                    facts.append(f"mentions {other.name}")
                continue
            facts.append(memory.describe_edge(e))
        out_nodes.append({"id": n.id, "type": n.type, "name": n.name if not n.is_episode else f"📄 {n.id}",
                          "summary": n.summary, "aliases": n.aliases, "confidence": round(n.confidence, 2),
                          "usage": n.usage_count, "facts": facts[:40], "pages": pages[:5]})
    out_edges = [{"s": e.src, "t": e.dst, "r": e.relation, "w": e.weight, "c": round(e.confidence, 2)}
                 for e in memory.edges.values() if e.src in keep and e.dst in keep]
    return {"nodes": out_nodes, "edges": out_edges, "stats": memory.stats()}


def render_html(memory: GraphMemory) -> str:
    data = json.dumps(graph_payload(memory), ensure_ascii=False).replace("</", "<\\/")
    return HTML.replace("__DATA__", data)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Kuni memory graph</title>
<style>
html,body{margin:0;height:100%;background:#12141a;color:#d8dbe6;font:14px system-ui,sans-serif}
#c{position:absolute;inset:0}
#panel{position:absolute;top:0;right:0;width:360px;max-height:100%;overflow:auto;background:#1b1e27ee;padding:12px;box-sizing:border-box}
#top{position:absolute;top:8px;left:8px;background:#1b1e27ee;padding:8px;border-radius:6px}
input{background:#0e1015;color:#d8dbe6;border:1px solid #3a3f52;padding:4px 6px;width:220px}
h3{margin:.2em 0}.f{margin:.3em 0;font-size:12px;color:#aeb4c8}.p{margin:.5em 0;padding:6px;background:#12141a;font-size:12px;white-space:pre-wrap}
.leg span{display:inline-block;margin-right:8px;font-size:12px}.leg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:3px}
</style></head><body><canvas id="c"></canvas>
<div id="top"><b>Kuni memory graph</b> <span id="st"></span><br><input id="q" placeholder="search…"><div class="leg" id="leg"></div></div>
<div id="panel"><i>click a node</i></div>
<script>
const D=__DATA__;const COL={person:"#ff8fab",place:"#7bd88f",org:"#ffd166",topic:"#6fb6ff",event:"#c792ea",object:"#f4a261",emotion:"#ef476f",self:"#ffffff",episode:"#5c6070"};
document.getElementById("st").textContent=D.stats.nodes+" nodes, "+D.stats.edges+" edges, "+D.stats.episodes+" pages";
document.getElementById("leg").innerHTML=Object.entries(COL).map(([k,v])=>`<span><i style="background:${v}"></i>${k}</span>`).join("");
const cv=document.getElementById("c"),cx=cv.getContext("2d");let W,H;function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight}rs();onresize=rs;
const N=D.nodes.map((n,i)=>({...n,x:W/2+Math.cos(i)*200*Math.random(),y:H/2+Math.sin(i)*200*Math.random(),vx:0,vy:0,deg:0}));
const ix={};N.forEach(n=>ix[n.id]=n);const E=D.edges.map(e=>({a:ix[e.s],b:ix[e.t],r:e.r,w:e.w,c:e.c})).filter(e=>e.a&&e.b);
E.forEach(e=>{e.a.deg++;e.b.deg++});let sel=null,hover=null,q="";
document.getElementById("q").oninput=e=>{q=e.target.value.toLowerCase()};
let tx=0,ty=0,sc=1,drag=null,pan=null;
function step(){const k=0.02;for(let i=0;i<N.length;i++){const a=N[i];for(let j=i+1;j<N.length;j++){const b=N[j];let dx=a.x-b.x,dy=a.y-b.y,d=dx*dx+dy*dy+0.01;if(d<90000){const f=2200/d;dx*=f;dy*=f;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy}}}
E.forEach(e=>{let dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01,f=(d-70)*0.01;dx/=d;dy/=d;e.a.vx+=dx*f;e.a.vy+=dy*f;e.b.vx-=dx*f;e.b.vy-=dy*f});
N.forEach(n=>{n.vx+=(W/2-n.x)*k*0.05;n.vy+=(H/2-n.y)*k*0.05;if(n!==drag){n.x+=n.vx*0.5;n.y+=n.vy*0.5}n.vx*=0.6;n.vy*=0.6})}
function R(n){return n.type=="episode"?3:4+Math.min(10,Math.sqrt(n.deg)*2)}
function draw(){step();cx.setTransform(1,0,0,1,0,0);cx.clearRect(0,0,W,H);cx.setTransform(sc,0,0,sc,tx,ty);
E.forEach(e=>{const hl=sel&&(e.a===sel||e.b===sel);cx.strokeStyle=hl?"#9aa5ff":"#2c3040";cx.lineWidth=hl?1.5:Math.min(3,0.5+e.w*0.3);cx.beginPath();cx.moveTo(e.a.x,e.a.y);cx.lineTo(e.b.x,e.b.y);cx.stroke();
if(hl&&e.r!="mentions"){cx.fillStyle="#9aa5ff";cx.font="10px sans-serif";cx.fillText(e.r,(e.a.x+e.b.x)/2,(e.a.y+e.b.y)/2)}});
N.forEach(n=>{const m=q&&(n.name.toLowerCase().includes(q)||n.aliases.join(" ").toLowerCase().includes(q)||n.summary.toLowerCase().includes(q));
cx.globalAlpha=q&&!m?0.15:1;cx.fillStyle=COL[n.type]||"#888";cx.beginPath();cx.arc(n.x,n.y,R(n),0,7);cx.fill();if(n===sel||n===hover){cx.strokeStyle="#fff";cx.lineWidth=2;cx.stroke()}
if(n.type!="episode"&&(n.deg>2||n===sel||n===hover||m)){cx.fillStyle="#d8dbe6";cx.font="11px sans-serif";cx.fillText(n.name,n.x+R(n)+2,n.y+3)}cx.globalAlpha=1});requestAnimationFrame(draw)}
function at(ev){const x=(ev.clientX-tx)/sc,y=(ev.clientY-ty)/sc;let b=null,bd=1e9;N.forEach(n=>{const d=Math.hypot(n.x-x,n.y-y);if(d<R(n)+4&&d<bd){b=n;bd=d}});return b}
cv.onmousedown=e=>{const n=at(e);if(n){drag=n;show(n)}else pan={x:e.clientX-tx,y:e.clientY-ty}};
cv.onmousemove=e=>{if(drag){drag.x=(e.clientX-tx)/sc;drag.y=(e.clientY-ty)/sc}else if(pan){tx=e.clientX-pan.x;ty=e.clientY-pan.y}else hover=at(e)};
onmouseup=()=>{drag=null;pan=null};cv.onwheel=e=>{e.preventDefault();const f=e.deltaY<0?1.1:0.9;tx=e.clientX-(e.clientX-tx)*f;ty=e.clientY-(e.clientY-ty)*f;sc*=f};
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function show(n){sel=n;document.getElementById("panel").innerHTML=`<h3>${esc(n.name)}</h3><div class="f">${n.type} · confidence ${n.confidence} · used ${n.usage}x</div>`+
(n.aliases.length?`<div class="f">aliases: ${esc(n.aliases.join(", "))}</div>`:"")+(n.summary?`<div class="p">${esc(n.summary)}</div>`:"")+
n.facts.map(f=>`<div class="f">• ${esc(f)}</div>`).join("")+n.pages.map(p=>`<div class="p">${esc(p)}</div>`).join("")}
draw();
</script></body></html>"""
