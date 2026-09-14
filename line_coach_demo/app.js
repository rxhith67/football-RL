"use strict";
const $ = id => document.getElementById(id);
const names = ["Normal play", "Low line", "High line"];
const descriptions = ["Native HELIOS decision-making.", "Off-ball defenders are directed toward a deeper line.", "Off-ball defenders are directed toward a higher line."];
let summary, replay, position = 0, playing = false, lastTime = 0, loadId = 0;
const colors = ["#d4f279", "#6bcbd1", "#d9a0e8"];
function message(error) { $("error").hidden = false; $("error").textContent = error; }
async function json(path) { const response = await fetch(path); if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`); return response.json(); }
function setPlaying(value) { playing = value; $("play").textContent = playing ? "Pause replay" : "Play replay"; }
async function loadReplay(filename) {
  const request = ++loadId; setPlaying(false);
  try {
    const data = await json(`data/${filename}`);
    if (request !== loadId) return;
    if (!data.frames?.length || data.frames.some(f => f.players?.length !== 22)) throw new Error("Invalid replay: expected 22 players in each frame");
    replay = data; position = 0; $("controller").textContent = data.controller; $("error").hidden = true;
    $("scrub").max = data.horizon;
    for (const id of ["play", "restart", "next", "scrub", "policy"]) $(id).disabled = false;
    draw();
  } catch (error) { message(`Could not load replay. ${error.message}`); }
}
function frameAt(step) {
  let lo = 0, hi = replay.frames.length - 1;
  while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (replay.frames[mid].step <= step) lo = mid; else hi = mid - 1; }
  const a = replay.frames[lo], b = replay.frames[Math.min(lo + 1, replay.frames.length - 1)];
  const t = b.step === a.step ? 0 : Math.max(0, Math.min(1, (step - a.step) / (b.step - a.step)));
  const interpolate = (p, q) => p.map((x, i) => x + (q[i] - x) * t);
  return {...a, ball: interpolate(a.ball, b.ball), players: a.players.map((p,i) => interpolate(p,b.players[i]))};
}
function draw() {
  const canvas = $("pitch"), ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  const margin = 42, scale = (w - margin * 2) / 105, left = margin, top = (h - 68 * scale) / 2;
  const point = p => [left + (p[0] + 52.5) * scale, top + (p[1] + 34) * scale];
  ctx.fillStyle = "#14372e"; ctx.fillRect(0,0,w,h);
  for (let i=0; i<10; i++) { ctx.fillStyle = i % 2 ? "#1b4135" : "#193c32"; ctx.fillRect(left + i*105*scale/10,top,105*scale/10,68*scale); }
  ctx.strokeStyle = "#9bbeb080"; ctx.lineWidth = 1.8;
  ctx.strokeRect(left,top,105*scale,68*scale);
  function line(a,b) { ctx.beginPath(); ctx.moveTo(...point(a)); ctx.lineTo(...point(b)); ctx.stroke(); }
  line([0,-34],[0,34]); ctx.beginPath(); ctx.arc(...point([0,0]),9.15*scale,0,Math.PI*2); ctx.stroke();
  for (const side of [-1,1]) {
    const x = side < 0 ? -52.5 : 36;
    ctx.strokeRect(...point([x,-20.16]),16.5*scale,40.32*scale);
    ctx.strokeRect(...point([side < 0 ? -52.5 : 47,-9.16]),5.5*scale,18.32*scale);
    ctx.strokeRect(...point([side < 0 ? -54 : 52.5,-7.01]),1.5*scale,14.02*scale);
  }
  if (!replay) return;
  const frame = frameAt(position);
  const lineX = frame.players.slice(1,5).reduce((s,p) => s+p[0],0)/4;
  ctx.save(); ctx.setLineDash([10,9]); ctx.strokeStyle="#d4f27a99"; ctx.lineWidth=2.5; line([lineX,-32],[lineX,32]); ctx.restore();
  frame.players.forEach((p,i) => {
    const xy = point(p); ctx.beginPath(); ctx.arc(...xy,10,0,Math.PI*2);
    ctx.fillStyle=i<11 ? "#d4f279" : "#6bcbd1"; ctx.shadowBlur=6; ctx.shadowColor="#0008"; ctx.fill(); ctx.shadowBlur=0;
    ctx.strokeStyle="#0b201d"; ctx.lineWidth=2; ctx.stroke(); ctx.fillStyle="#132a25";
    ctx.font="bold 10px system-ui"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(String(i%11+1),xy[0],xy[1]+.4);
  });
  ctx.beginPath(); ctx.arc(...point(frame.ball),5.5,0,Math.PI*2); ctx.fillStyle="#fff"; ctx.fill(); ctx.strokeStyle="#152925"; ctx.lineWidth=2; ctx.stroke();
  const decisions = replay.decisions.filter(d => d.step <= position), current = decisions.at(-1)?.action ?? 0;
  $("instruction").textContent=names[current]; $("instruction-detail").textContent=descriptions[current];
  $("home-score").textContent=frame.score[0]; $("away-score").textContent=frame.score[1];
  $("clock").textContent=`${(position/10).toFixed(1)} s action time`;
  $("line").textContent=`${lineX.toFixed(1)} m`; $("stamina").textContent=`${(frame.stamina*100).toFixed(0)}%`;
  $("step").textContent=`${Math.floor(position)} / ${replay.horizon} native steps`; $("scrub").value=position;
  const log = $("decisions"); log.replaceChildren();
  decisions.slice(-6).reverse().forEach((d,i) => { const li=document.createElement("li"); if (!i) li.className="active";
    const time=document.createElement("span"), label=document.createElement("span"); time.textContent=`Step ${d.step}`; label.textContent=names[d.action]; li.append(time,label); log.append(li); });
}
function trainingChart(training) {
  const ctx=$("learning").getContext("2d"), w=800,h=300, pad=40;
  ctx.clearRect(0,0,w,h); const curves=training.map(t=>t.episodes.map((e,i,es)=>({x:e.timesteps,y:es.slice(Math.max(0,i-4),i+1).reduce((s,v)=>s+v.r,0)/Math.min(i+1,5)})));
  const values=curves.flat().map(v=>v.y); if (!values.length) return;
  let min=Math.min(...values),max=Math.max(...values); if(max-min<.1){min-=.1;max+=.1;}
  const xmax=Math.max(...training.map(t=>t.timesteps));
  ctx.font="11px system-ui"; ctx.textAlign="right";
  for(let i=0;i<=4;i++){const y=pad+i*(h-2*pad)/4;ctx.strokeStyle="#293a43";ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke();ctx.fillStyle="#98aaae";ctx.fillText((max-i*(max-min)/4).toFixed(2),pad-8,y+4);}
  curves.forEach((points,i)=>{ctx.strokeStyle=colors[i%3];ctx.lineWidth=2;ctx.beginPath();points.forEach((p,j)=>{const x=pad+p.x/xmax*(w-2*pad),y=h-pad-(p.y-min)/(max-min)*(h-2*pad);if(j)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();});
  ctx.fillStyle="#98aaae";ctx.textAlign="left";ctx.fillText("0 decisions",pad,h-10);ctx.textAlign="right";ctx.fillText(`${xmax} decisions`,w-pad,h-10);
  $("legend").replaceChildren();training.forEach((t,i)=>{const item=document.createElement("span");item.style.color=colors[i%3];item.textContent=`● PPO seed ${t.seed}`;$("legend").append(item)});
}
async function boot() {
  try {
    summary=await json("data/summary.json");
    $("episode-count").textContent=`${summary.evaluation_episodes} HELD-OUT EPISODES`;
    for(const row of summary.results){
      const tr=document.createElement("tr"); if(row.policy==="ppo")tr.className="ppo";
      [row.name,row.episodes,`${row.wins} / ${row.draws} / ${row.losses}`,row.mean_goal_difference.toFixed(2),row.goal_difference_ci95.map(v=>v.toFixed(2)).join(" to "),row.points_per_episode.toFixed(2)].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td)});
      $("results-body").append(tr);
      if(row.replay){const option=document.createElement("option");option.value=row.replay;option.textContent=row.name;$("policy").append(option);}
    }
    const ppo=summary.results.filter(r=>r.policy==="ppo"), baselines=summary.results.filter(r=>r.policy!=="ppo");
    const mean=ppo.reduce((s,r)=>s+r.mean_goal_difference,0)/ppo.length,best=baselines.reduce((a,b)=>a.mean_goal_difference>b.mean_goal_difference?a:b);
    $("conclusion").textContent=`Across training seeds, PPO averaged ${mean.toFixed(2)} goal difference; the strongest baseline (${best.name}) averaged ${best.mean_goal_difference.toFixed(2)}. ${mean>best.mean_goal_difference?"The observed PPO mean is higher, but this small study does not establish a reliable advantage.":"This experiment does not demonstrate an advantage over the strongest baseline."}`;
    const singleAction=ppo.filter(r=>r.actions.filter(n=>n>0).length===1);
    $("behaviour").textContent=singleAction.length ? `Policy behaviour: ${singleAction.map(r=>`${r.name} chose ${names[r.actions.findIndex(n=>n>0)]} for every evaluated decision`).join('; ')}. These policies did not demonstrate adaptive switching.` : "The learned policies used multiple instructions during evaluation. Instruction usage alone does not establish tactical effectiveness.";
    for(const text of summary.limitations){const li=document.createElement("li");li.textContent=text;$("limitations").append(li);}
    trainingChart(summary.training);
    const first=summary.results.find(r=>r.policy==="ppo"&&r.seed===summary.selected_demo_seed);
    if(first?.replay){$("policy").value=first.replay;await loadReplay(first.replay);}
  }catch(error){message(`Experiment data is unavailable. Run the publisher after training, then start the demo with python scripts/serve_line_coach.py. ${error.message}`);}
}
$("play").addEventListener("click",()=>{if(position>=replay.horizon)position=0;setPlaying(!playing)});
$("restart").addEventListener("click",()=>{position=0;setPlaying(false);draw()});
$("scrub").addEventListener("input",e=>{position=Number(e.target.value);setPlaying(false);draw()});
$("next").addEventListener("click",()=>{setPlaying(false);position=replay.decisions.find(d=>d.step>position)?.step??replay.horizon;draw()});
$("policy").addEventListener("change",e=>loadReplay(e.target.value));
function animate(time){if(playing&&replay&&lastTime){position=Math.min(replay.horizon,position+Math.min(time-lastTime,100)/100*Number($("speed").value));draw();if(position>=replay.horizon)setPlaying(false)}lastTime=time;requestAnimationFrame(animate)}
draw();boot();requestAnimationFrame(animate);
