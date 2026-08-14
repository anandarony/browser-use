"""Web UI for browser-use with LIVE, per-step lifecycle progress.

Launched by `npm run dev` (see package.json). Backend is Azure OpenAI when
configured in .env, otherwise the local Ollama model. Server binds to BROWSER_PORT.

    npm run dev        ->  http://localhost:7780

Per-run options (chosen in the UI, sent per run — never persisted):
  Viewport   desktop (maximized window) | mobile (emulated phone)
  User type  non-sso -> use the app's own username/password login form.
             sso     -> click the "Single sign on" button and authenticate on the
             identity-provider page instead. When SSO, an optional authenticator
             secret (TOTP seed) can be entered; it is sent in the POST body, used
             in-memory for that single run as browser-use `sensitive_data` (a
             `bu_2fa_code` placeholder -> live 6-digit code), filtered from logs,
             and never written to disk/.env. One-time read only.

Each agent step streams three events so the UI is never a black box:
  step_start  -> a card appears the moment the step begins (before the LLM call)
  step        -> the card fills in with the model's goal / assessment / actions
  step_end    -> the card flips to done/error with the step's actual result
"""

import asyncio
import json
import os
import re

from aiohttp import web
from dotenv import load_dotenv

from browser_use import Agent, BrowserProfile, ChatAzureOpenAI, ChatOllama
from browser_use.llm.messages import SystemMessage, UserMessage

load_dotenv()

PORT = int(os.getenv('BROWSER_PORT', '7780'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
MAX_STEPS = int(os.getenv('AGENT_MAX_STEPS', '25'))

# Azure OpenAI (ITM) config — if a key is present we use Azure instead of Ollama.
AZURE_API_KEY = os.getenv('ITM_AI_AZURE_OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('ITM_AI_AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_DEPLOYMENT = os.getenv('ITM_AI_AZURE_OPENAI_DEPLOYMENT') or os.getenv('AZURE_OPENAI_DEPLOYMENT') or 'gpt-4.1'
AZURE_API_VERSION = os.getenv('ITM_AI_AZURE_OPENAI_API_VERSION') or '2025-01-01-preview'
USE_AZURE = bool(AZURE_API_KEY and AZURE_ENDPOINT)

# What the UI badge shows and which backend the agent runs on.
BACKEND_LABEL = f'azure · {AZURE_DEPLOYMENT}' if USE_AZURE else OLLAMA_MODEL
BACKEND_HOST = AZURE_ENDPOINT if USE_AZURE else OLLAMA_HOST


# Browser rendering/timing — SPAs (client-side rendered) return a blank DOM if
# snapshotted too early, which makes the agent loop on "page is blank -> refresh".
# Headful + generous load waits let the app hydrate before we capture state.
BROWSER_HEADLESS = os.getenv('BROWSER_USE_HEADLESS', 'false').lower() in ('1', 'true', 'yes')
PAGE_MIN_WAIT = float(os.getenv('PAGE_MIN_WAIT', '2.0'))
PAGE_NETWORK_IDLE_WAIT = float(os.getenv('PAGE_NETWORK_IDLE_WAIT', '3.0'))

MOBILE_USER_AGENT = (
	'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 '
	'(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
)

# Applied to every run so the agent identifies the correct element before acting,
# instead of typing into / clicking the wrong field.
EXECUTION_RULES = (
	'\n\n[EXECUTION RULES — analyse every field before acting]\n'
	'1. Before typing or clicking, examine ALL interactive elements currently on the page '
	'(inputs, buttons, links, dropdowns) and read each one\'s visible label, placeholder text, '
	'and the text right next to it.\n'
	'2. Type a value ONLY into the field whose label/placeholder matches its purpose: the '
	'username / login id goes in the field labelled username, email, or login — NOT a search box, '
	'the URL bar, or any unrelated field; the password goes ONLY in the password field.\n'
	'3. Do NOT click navigation links, menu items, or buttons that are unrelated to the current step. '
	'Match the exact button/link text the step asks for.\n'
	'4. After typing, verify the value actually appears in the intended field. If it went into the '
	'wrong field, clear it and correct it before continuing.\n'
	'5. If the field or control a step needs is not present on the page, do NOT force the action onto '
	'a similar-looking element — report that it was not found for that step.'
)


def build_browser_profile(device: str = 'desktop') -> BrowserProfile:
	"""Profile tuned so single-page apps finish rendering before DOM capture.

	device='desktop' -> maximized full window (Chrome --start-maximized when headful)
	device='mobile'  -> emulated iPhone-sized viewport + mobile user agent
	"""
	common = dict(
		headless=BROWSER_HEADLESS,
		minimum_wait_page_load_time=PAGE_MIN_WAIT,
		wait_for_network_idle_page_load_time=PAGE_NETWORK_IDLE_WAIT,
	)
	if device == 'mobile':
		return BrowserProfile(
			**common,
			window_size={'width': 390, 'height': 844},
			viewport={'width': 390, 'height': 844},
			device_scale_factor=3.0,
			user_agent=MOBILE_USER_AGENT,
		)
	# desktop: window_size=None + headful => Chrome launches with --start-maximized
	return BrowserProfile(**common, window_size=None, no_viewport=True)


def build_llm():
	"""Return the chat model: Azure OpenAI when configured, else local Ollama."""
	if USE_AZURE:
		return ChatAzureOpenAI(
			model=AZURE_DEPLOYMENT,
			api_key=AZURE_API_KEY,
			azure_endpoint=AZURE_ENDPOINT,
			azure_deployment=AZURE_DEPLOYMENT,
			api_version=AZURE_API_VERSION,
		)
	return ChatOllama(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=300.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _summarize_actions(model_output) -> list[dict]:
	out: list[dict] = []
	for action in getattr(model_output, 'action', None) or []:
		try:
			dumped = action.model_dump(exclude_none=True)
		except Exception:
			continue
		if not dumped:
			continue
		name = next(iter(dumped))
		params = dumped[name]
		if isinstance(params, dict):
			params = ', '.join(f'{k}={v!r}' for k, v in list(params.items())[:4])
		out.append({'name': name, 'params': str(params)})
	return out


def _cached_url(agent) -> str:
	try:
		summ = agent.browser_session._cached_browser_state_summary
		if summ and getattr(summ, 'url', None):
			return summ.url
	except Exception:
		pass
	return ''


async def _live_url(agent) -> str:
	try:
		return await agent.browser_session.get_current_page_url()
	except Exception:
		return _cached_url(agent)


STEP_LINE_RE = re.compile(r'^\s*(\d+)[.)]\s+(\S.*)$', re.M)
TITLE_RE = re.compile(r'^\s*Title:\s*(.+)$', re.M | re.I)
STEP_TAG_RE = re.compile(r'\[STEP\s*(\d+)\]', re.I)


def parse_testcase_steps(text: str) -> list[dict]:
	"""Extract the numbered steps ("1. ...", "2. ...") from a test case for the checklist."""
	return [{'n': int(m.group(1)), 'text': m.group(2).strip()} for m in STEP_LINE_RE.finditer(text)]


def parse_testcase_title(text: str) -> str:
	m = TITLE_RE.search(text)
	return m.group(1).strip() if m else ''


def _extract_step(model_output) -> int | None:
	"""Read the [STEP N] tag the model is asked to emit, so we know which test step is active."""
	for attr in ('next_goal', 'memory', 'thinking'):
		v = getattr(model_output, attr, None)
		if v:
			m = STEP_TAG_RE.search(str(v))
			if m:
				return int(m.group(1))
	return None


def _summarize_results(agent) -> list[dict]:
	"""Turn the step's ActionResult list into compact status rows for the UI."""
	rows: list[dict] = []
	for r in getattr(agent.state, 'last_result', None) or []:
		if getattr(r, 'error', None):
			rows.append({'status': 'error', 'text': str(r.error)[:300]})
		elif getattr(r, 'extracted_content', None):
			rows.append({'status': 'ok', 'text': str(r.extracted_content)[:300]})
		elif getattr(r, 'long_term_memory', None):
			rows.append({'status': 'ok', 'text': str(r.long_term_memory)[:300]})
		elif getattr(r, 'is_done', None):
			rows.append({'status': 'done', 'text': 'task complete'})
	return rows


# ---------------------------------------------------------------------------
# HTML (single page app)
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA Automation</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#141a24; --panel2:#1b2431; --line:#263141;
    --text:#e6edf3; --muted:#8b98a9; --accent:#4f8cff; --ok:#22c55e;
    --warn:#f59e0b; --err:#ef4444; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
  header{position:sticky;top:0;z-index:5;background:rgba(11,14,20,.85);backdrop-filter:blur(8px);
         border-bottom:1px solid var(--line);padding:14px 22px;display:flex;align-items:center;gap:14px}
  header h1{font-size:16px;margin:0;font-weight:650}
  .badge{font-family:var(--mono);font-size:12px;color:var(--muted);background:var(--panel2);
         border:1px solid var(--line);padding:3px 9px;border-radius:999px}
  .pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;
        padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--panel2)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
  .pill.run .dot{background:var(--accent);animation:pulse 1s infinite}
  .pill.done .dot{background:var(--ok)} .pill.err .dot{background:var(--err)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
  main{max-width:920px;margin:0 auto;padding:22px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:18px}
  label{display:block;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
  textarea{width:100%;min-height:120px;resize:vertical;background:var(--bg);color:var(--text);
           border:1px solid var(--line);border-radius:8px;padding:12px;font-size:14px;font-family:var(--mono)}
  .opts{display:flex;flex-wrap:wrap;gap:18px;margin-top:14px}
  .opt{min-width:200px}
  .seg{display:flex;gap:8px}
  .chip{display:flex;align-items:center;gap:7px;cursor:pointer;text-transform:none;letter-spacing:0;
        font-size:13px;color:var(--text);background:var(--bg);border:1px solid var(--line);
        border-radius:8px;padding:8px 12px;margin:0}
  .chip:has(input:checked){border-color:var(--accent);background:rgba(79,140,255,.12);color:#cfe0ff}
  .chip input{accent-color:var(--accent);margin:0}
  .chip .sub{color:var(--muted);font-size:11px}
  #secretWrap input{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);
        border-radius:8px;padding:10px 12px;font-size:14px;font-family:var(--mono)}
  #secretWrap .sub{text-transform:none;letter-spacing:0;color:var(--warn);font-size:11px;font-weight:600}
  .row{display:flex;gap:10px;align-items:center;margin-top:16px}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer}
  button.secondary{background:var(--panel2);border:1px solid var(--line);color:var(--text)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .hint{color:var(--muted);font-size:12px;margin-left:auto}
  h2.sec{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin:6px 2px 12px;display:flex;gap:10px;align-items:center}
  #steps{display:flex;flex-direction:column;gap:12px}
  .step{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);border-radius:10px;padding:12px 14px;animation:slidein .25s ease}
  .step.running{border-left-color:var(--accent);box-shadow:0 0 0 1px rgba(79,140,255,.2)}
  .step.done{border-left-color:var(--ok)} .step.error{border-left-color:var(--err)}
  @keyframes slidein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .step .head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .num{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--accent);background:var(--panel2);
       border:1px solid var(--line);border-radius:6px;padding:2px 8px}
  .st{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:2px 8px;border-radius:999px;display:flex;align-items:center;gap:6px}
  .st.running{color:#bcd4ff;background:rgba(79,140,255,.12);border:1px solid rgba(79,140,255,.4)}
  .st.done{color:#bbf7d0;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.4)}
  .st.error{color:#fecaca;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.4)}
  .goal{font-weight:600;font-size:14px;width:100%;margin-top:2px}
  .spin{display:inline-block;vertical-align:middle;width:12px;height:12px;border:2px solid rgba(255,255,255,.25);border-top-color:#bcd4ff;border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .kv{font-size:13px;color:var(--muted);margin:3px 0}
  .kv b{color:var(--text);font-weight:600}
  .acts{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .act{font-family:var(--mono);font-size:12px;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:3px 8px}
  .act .p{color:var(--muted)}
  .url{font-family:var(--mono);font-size:12px;color:var(--accent);word-break:break-all}
  .res{font-size:13px;margin-top:8px;padding:8px 10px;border-radius:8px;background:var(--panel2);border:1px solid var(--line)}
  .res.ok{border-left:3px solid var(--ok)} .res.error{border-left:3px solid var(--err);color:#fecaca}
  #result{white-space:pre-wrap;font-size:14px} #result.ok{border-left:3px solid var(--ok)} #result.bad{border-left:3px solid var(--err);color:#fecaca}
  #shot img{max-width:100%;border-radius:8px;border:1px solid var(--line);margin-top:6px}
  .empty{color:var(--muted);font-size:13px;text-align:center;padding:22px}
  .counter,.mode{font-family:var(--mono);font-size:12px;color:var(--muted)}
  /* animated test-case checklist */
  .tc{display:flex;gap:12px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);
      border-left:3px solid var(--line);border-radius:10px;padding:11px 14px;
      transition:border-color .3s,background .3s;animation:slidein .25s ease}
  .tc.running{border-left-color:var(--accent);background:rgba(79,140,255,.06)}
  .tc.done{border-left-color:var(--ok)} .tc.failed{border-left-color:var(--err)}
  .tcicon{flex:0 0 24px;width:24px;height:24px;display:flex;align-items:center;justify-content:center;
          border-radius:50%;font-size:13px;font-weight:700;background:var(--panel2);
          border:1px solid var(--line);color:var(--muted)}
  .tc.running .tcicon{color:#bcd4ff;border-color:rgba(79,140,255,.6);background:rgba(79,140,255,.15);
          font-size:16px;font-weight:800;animation:arrow 1s ease-in-out infinite}
  .tc.done .tcicon{color:#052e16;background:var(--ok);border-color:var(--ok);animation:pop .3s ease}
  .tc.failed .tcicon{color:#fff;background:var(--err);border-color:var(--err);animation:pop .3s ease}
  @keyframes pop{0%{transform:scale(.4);opacity:.2}60%{transform:scale(1.18)}100%{transform:scale(1);opacity:1}}
  @keyframes arrow{0%,100%{transform:translateX(-2px)}50%{transform:translateX(3px)}}
  .tcstat{flex:0 0 auto;align-self:center;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
          color:var(--muted);padding:3px 11px;border-radius:999px;border:1px solid var(--line);background:var(--panel2);white-space:nowrap}
  .tc.running .tcstat{color:#bcd4ff;border-color:rgba(79,140,255,.5);background:rgba(79,140,255,.12)}
  .tc.done .tcstat{color:#bbf7d0;border-color:rgba(34,197,94,.4);background:rgba(34,197,94,.12)}
  .tc.failed .tcstat{color:#fecaca;border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.12)}
  .tcbody{flex:1;min-width:0}
  .tctext{font-size:14px}
  .tctext b{color:var(--accent);font-family:var(--mono);margin-right:4px}
  .tc.done .tctext{color:var(--muted)}
  .tcdetail{font-size:12px;color:var(--muted);margin-top:5px;font-family:var(--mono);display:none;word-break:break-word}
  .tc.running .tcdetail{display:block}
  .tc.failed .tcdetail{display:block;color:#fecaca}
  .tcdetail .u{color:var(--accent)}
</style></head><body>
<header>
  <h1>QA Automation</h1>
  <span class="pill" id="status"><span class="dot"></span><span id="statusText">Idle</span></span>
</header>
<main>
  <div class="card">
    <label for="desc">Describe your test</label>
    <textarea id="desc" placeholder="Describe the application, feature, test scenario, expected behaviour, test data, browser/device requirements, and any specific validation rules. The more details you provide, the more accurate and comprehensive the generated automation test cases will be."></textarea>

    <div class="opts">
      <div class="opt">
        <label>Viewport</label>
        <div class="seg">
          <label class="chip"><input type="radio" name="device" value="desktop" checked onchange="onOpts()"> Desktop <span class="sub">maximized</span></label>
          <label class="chip"><input type="radio" name="device" value="mobile" onchange="onOpts()"> Mobile <span class="sub">phone view</span></label>
        </div>
      </div>
      <div class="opt">
        <label>User type</label>
        <div class="seg">
          <label class="chip"><input type="radio" name="usertype" value="non-sso" checked onchange="onOpts()"> Non-SSO</label>
          <label class="chip"><input type="radio" name="usertype" value="sso" onchange="onOpts()"> SSO</label>
        </div>
      </div>
      <div class="opt" id="secretWrap" style="display:none;flex:1 1 100%">
        <label for="secret">Authenticator secret <span class="sub">TOTP seed — read once, never stored</span></label>
        <input id="secret" type="password" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="e.g. JBSWY3DPEHPK3PXP (optional)">
      </div>
    </div>

    <div class="row">
      <button id="create" onclick="createTask()">Create Task</button>
      <span class="hint" id="createHint"></span>
    </div>
  </div>

  <div class="card" id="tcCard" style="display:none">
    <label for="task">Standard test case &mdash; review &amp; edit before running</label>
    <textarea id="task" style="min-height:210px" placeholder="The generated test case will appear here. Edit it freely, then Start Testing."></textarea>

    <div class="row">
      <button id="go" onclick="run()">Start Testing</button>
      <button id="stop" class="secondary" onclick="stop()" disabled>Stop</button>
      <button id="regen" class="secondary" onclick="createTask()">Regenerate</button>
      <span class="hint" id="timer"></span>
    </div>
  </div>

  <h2 class="sec">Live progress <span class="mode" id="mode"></span> <span class="counter" id="counter"></span></h2>
  <div id="steps"><div class="empty" id="stepsEmpty">No run yet. Enter a task and press <b>Run task</b>.</div></div>

  <div id="finalWrap" style="display:none">
    <h2 class="sec">Result</h2>
    <div class="card" id="result"></div>
    <div id="shot"></div>
  </div>
</main>
<script>
let ctrl=null, t0=0, timerId=null, tcEls={}, tcCount=0, curStep=0, stepError={};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const picked=name=>document.querySelector(`input[name="${name}"]:checked`).value;

function setStatus(cls,text){$('status').className='pill '+cls;$('statusText').textContent=text;}
function fmt(s){const m=Math.floor(s/60),ss=s%60;return m?`${m}m ${ss}s`:`${ss}s`;}
function startTimer(){t0=Date.now();timerId=setInterval(()=>{$('timer').textContent='elapsed '+fmt(Math.floor((Date.now()-t0)/1000));},1000);}
function stopTimer(){clearInterval(timerId);}
function onOpts(){ $('secretWrap').style.display = (picked('usertype')==='sso') ? 'block' : 'none'; }

// ---- animated test-case checklist ----
function countDone(){ let c=0; for(let i=1;i<=tcCount;i++){ const e=tcEls[i]; if(e&&e.classList.contains('done')) c++; } return c; }
function setCounter(){ $('counter').textContent = tcCount?`${countDone()}/${tcCount} steps`:''; }
function initChecklist(steps){
  $('steps').innerHTML=''; tcEls={}; tcCount=steps.length; curStep=0; stepError={};
  if(!tcCount){ $('steps').innerHTML='<div class="empty">Running… (no numbered steps found in the test case)</div>'; setCounter(); return; }
  steps.forEach(s=>{
    const el=document.createElement('div'); el.className='tc pending'; el.id='tc-'+s.n;
    el.innerHTML=`<span class="tcicon" id="tcicon-${s.n}">${s.n}</span>`
      +`<div class="tcbody"><div class="tctext"><b>${s.n}.</b> ${esc(s.text)}</div>`
      +`<div class="tcdetail" id="tcdetail-${s.n}"></div></div>`
      +`<span class="tcstat" id="tcstat-${s.n}">Pending</span>`;
    $('steps').appendChild(el); tcEls[s.n]=el;
  });
  setCounter();
}
function setStep(n,state){
  const el=tcEls[n]; if(!el) return; el.className='tc '+state;
  const ic=$('tcicon-'+n), stt=$('tcstat-'+n);
  if(ic){
    if(state==='running') ic.textContent='→';
    else if(state==='done') ic.textContent='✓';
    else if(state==='failed') ic.textContent='✕';
    else ic.textContent=n;
  }
  if(stt){
    stt.textContent = state==='running'?'Running':state==='done'?'Done':state==='failed'?'Failed':'Pending';
  }
}
function activate(n){
  if(!tcCount) return;
  n=Math.max(1,Math.min(n,tcCount));
  // steps we've moved past are done — unless they ended in an error, then failed
  for(let i=1;i<n;i++){ if(tcEls[i]) setStep(i, stepError[i] ? 'failed' : 'done'); }
  setStep(n,'running'); curStep=Math.max(curStep,n);
  setCounter(); tcEls[n]?.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function detail(n,html){ const d=$('tcdetail-'+n); if(d) d.innerHTML=html; }
function markAllDone(){ for(let i=1;i<=tcCount;i++){ if(tcEls[i]) setStep(i, stepError[i] ? 'failed' : 'done'); } setCounter(); }
function resetProgress(){
  stopTimer();
  tcEls={}; tcCount=0; curStep=0; stepError={};
  $('steps').innerHTML='<div class="empty" id="stepsEmpty">No run yet. Review the test case, then Start Testing.</div>';
  $('counter').textContent=''; $('mode').textContent=''; $('timer').textContent='';
  $('finalWrap').style.display='none'; $('result').textContent=''; $('result').className='card'; $('shot').innerHTML='';
  setStatus('idle','Idle');
}

function dispatch(ev,data){
  let d={}; try{ d=data?JSON.parse(data):{}; }catch(e){ return; }
  if(ev==='tc_init'){ $('stepsEmpty')?.remove(); initChecklist(d.steps||[]); }
  else if(ev==='tc_active'){
    // on_step_end sends results — record whether THIS step ended in an error (last state wins)
    if(Array.isArray(d.results) && d.step){
      const errs=d.results.filter(r=>r.status==='error');
      stepError[d.step]=errs.length>0;
      if(errs.length) detail(d.step, '✕ '+errs.map(r=>esc(r.text)).join(' · '));
    }
    if(d.step) activate(d.step);
    if(!Array.isArray(d.results)){
      const acts=(d.actions||[]).map(a=>esc(a.name)).join(', ');
      const parts=[]; if(d.goal) parts.push(esc(d.goal)); if(acts) parts.push('· '+acts);
      if(d.url) parts.push('· <span class="u">'+esc(d.url)+'</span>');
      if(d.step) detail(d.step, parts.join(' '));
    }
    setStatus('run','Running · step '+(d.step||curStep||'?')+(tcCount?('/'+tcCount):''));
  }
  else if(ev==='meta'){ $('mode').textContent = `${d.device} · ${d.sso?'SSO':'Non-SSO'}${d.secret?' + 2FA':''}`; }
  else if(ev==='done'){ finish(); markAllDone();
    const failed=Object.values(stepError).filter(Boolean).length;
    if(failed) setStatus('err','Done · '+failed+' failed'); else setStatus('done','Done · '+countDone()+(tcCount?('/'+tcCount):'')+' steps');
    $('finalWrap').style.display='block';
    $('result').className=failed?'card bad':'card ok'; $('result').textContent=d.result||'(agent returned no final text)';
    if(d.screenshot) $('shot').innerHTML=`<img alt="final screenshot" src="data:image/png;base64,${d.screenshot}">`; }
  else if(ev==='error_msg'){ finish(); if(curStep&&tcEls[curStep]) setStep(curStep,'failed'); setStatus('err','Error');
    $('finalWrap').style.display='block';
    $('result').className='card bad'; $('result').textContent=d.error||'unknown error'; }
}

async function createTask(){
  const desc=$('desc').value.trim(); if(!desc){alert('Describe your test first');return;}
  resetProgress();  // clear any previous run's live progress + result
  const btn=$('create'), regen=$('regen');
  btn.disabled=true; if(regen) regen.disabled=true;
  $('createHint').innerHTML='<span class="spin"></span> Generating standard test case…';
  try{
    const r=await fetch('/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({description:desc})});
    const j=await r.json();
    if(!j.ok){ $('createHint').textContent='Error: '+(j.error||'failed to create test case'); }
    else{
      $('task').value=j.testcase||'';
      $('tcCard').style.display='block';
      $('createHint').textContent='Test case created — review/edit it below, then Start Testing.';
      $('tcCard').scrollIntoView({behavior:'smooth',block:'start'});
    }
  }catch(e){ $('createHint').textContent='Request failed: '+e; }
  btn.disabled=false; if(regen) regen.disabled=false;
}

async function run(){
  const task=$('task').value.trim(); if(!task){alert('Create a task first, or edit the test case before running');return;}
  const body={ task, device:picked('device'), usertype:picked('usertype'), secret: picked('usertype')==='sso' ? $('secret').value : '' };
  // one-time read: clear the secret field from the DOM immediately after capturing it
  $('secret').value='';
  $('go').disabled=true; $('stop').disabled=false;
  $('steps').innerHTML='<div class="empty">Preparing test case…</div>'; tcEls={}; tcCount=0; curStep=0; stepError={}; $('counter').textContent='';
  $('finalWrap').style.display='none'; $('shot').innerHTML=''; $('mode').textContent='';
  setStatus('run','Starting…'); startTimer();
  ctrl=new AbortController();
  try{
    const resp=await fetch('/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    // scrub the secret from the JS object as soon as it's on the wire
    body.secret='';
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
    for(;;){
      const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let i; while((i=buf.indexOf('\n\n'))>=0){
        const chunk=buf.slice(0,i); buf=buf.slice(i+2);
        let ev='message',data='';
        for(const line of chunk.split('\n')){
          if(line.startsWith('event:')) ev=line.slice(6).trim();
          else if(line.startsWith('data:')) data+=line.slice(5).trim();
        }
        dispatch(ev,data);
      }
    }
    if($('go').disabled){ finish(); setStatus('done','Stream ended'); }
  }catch(e){
    finish();
    setStatus('err', e.name==='AbortError' ? 'Stopped' : 'Disconnected');
  }
}
function finish(){ if(ctrl){try{ctrl.abort();}catch(e){} ctrl=null;} stopTimer(); $('go').disabled=false; $('stop').disabled=true; }
function stop(){ finish(); setStatus('idle','Stopped'); }
onOpts();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# test-case generation
# ---------------------------------------------------------------------------
TESTCASE_SYSTEM = (
	'You are a senior QA automation engineer. Convert the user\'s description into ONE standard '
	'manual test case that a browser-automation agent can execute step by step.\n\n'
	'Output PLAIN TEXT (no markdown, no code fences) in exactly this shape:\n'
	'Title: <short title>\n'
	'Preconditions: <setup / test data, or "None">\n'
	'Steps:\n'
	'1. <one atomic action per line: navigate / click / type / select / verify>\n'
	'2. ...\n'
	'Expected Result: <the final expected outcome>\n\n'
	'Rules: one concrete action per numbered step. Use the EXACT test data (URLs, usernames, '
	'values) the user gave — never invent credentials or URLs. If the user omitted details, make '
	'the steps generic rather than guessing specifics. Keep it concise and directly executable.'
)


async def generate_testcase(description: str) -> str:
	llm = build_llm()
	res = await llm.ainvoke([SystemMessage(content=TESTCASE_SYSTEM), UserMessage(content=description)])
	return (res.completion or '').strip()


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
async def handle_create(request: web.Request) -> web.Response:
	try:
		payload = await request.json()
	except Exception:
		payload = {}
	description = (payload.get('description') or '').strip()
	if not description:
		return web.json_response({'ok': False, 'error': 'empty description'}, status=400)
	try:
		testcase = await generate_testcase(description)
		return web.json_response({'ok': True, 'testcase': testcase})
	except Exception as e:
		return web.json_response({'ok': False, 'error': str(e)}, status=500)


async def handle_index(request: web.Request) -> web.Response:
	return web.Response(text=PAGE, content_type='text/html')


async def handle_stream(request: web.Request) -> web.StreamResponse:
	try:
		payload = await request.json()
	except Exception:
		payload = {}
	task = (payload.get('task') or '').strip()
	device = 'mobile' if payload.get('device') == 'mobile' else 'desktop'
	is_sso = payload.get('usertype') == 'sso'
	secret = (payload.get('secret') or '').strip()  # in-memory only; never logged or persisted

	resp = web.StreamResponse(
		status=200,
		headers={
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
			'X-Accel-Buffering': 'no',
		},
	)
	await resp.prepare(request)

	async def send(event: str, data: dict) -> None:
		await resp.write(f'event: {event}\ndata: {json.dumps(data)}\n\n'.encode())

	if not task:
		await send('error_msg', {'error': 'empty task'})
		return resp

	await send('meta', {'device': device, 'sso': is_sso, 'secret': bool(is_sso and secret)})

	# Parse the numbered test-case steps for the live checklist BEFORE appending directives.
	tc_steps = parse_testcase_steps(task)
	await send('tc_init', {'steps': tc_steps, 'title': parse_testcase_title(task)})

	# Branch the LOGIN METHOD by user type. This is what was missing before: SSO
	# must take the "Single sign on" route, not the app's own username/password form.
	sensitive_data = None
	if is_sso:
		directive = (
			'\n\n[LOGIN METHOD: SSO — Single Sign-On]\n'
			"Do NOT type into the application's own email/username or password fields, and do "
			"NOT use the 'Login with code' button. Instead, click the 'Single sign on' (SSO) "
			'button and complete authentication on the identity-provider page it redirects to. '
			'If that provider page asks for credentials and the task specified a username/password, '
			'enter them there.'
		)
		if secret:
			# A placeholder name ending in `bu_2fa_code` makes browser-use generate a live
			# TOTP code from the seed. The raw seed stays in memory for this run only and is
			# filtered from logs (browser-use sensitive_data handling).
			sensitive_data = {'sso_bu_2fa_code': secret}
			directive += (
				' If a one-time authenticator / 2FA code is required, type the placeholder '
				'<secret>sso_bu_2fa_code</secret> into the code field — it is replaced with the '
				'current 6-digit code.'
			)
		task += directive
	else:
		task += (
			'\n\n[LOGIN METHOD: standard]\n'
			'Log in using the application\'s own email/username and password form. '
			'Do NOT use the "Single sign on" (SSO) option.'
		)

	# Precise element handling — analyse each field before acting (avoid wrong-field typing/clicks).
	task += EXECUTION_RULES

	# Ask the model to tag which numbered step it is on, so the checklist can animate.
	if tc_steps:
		task += (
			'\n\n[PROGRESS REPORTING]\n'
			'Execute the numbered steps strictly in order. At the very START of every next_goal, '
			'output the tag [STEP N] where N is the exact numbered test-case step you are currently '
			'performing (e.g. "[STEP 2] type the username"). Always include this tag.'
		)

	queue: asyncio.Queue = asyncio.Queue()
	current = {'step': 1}  # which test-case step is active (from the model's [STEP N] tag)

	async def on_step_start(agent) -> None:
		# show the current step as running immediately, before the (slow) LLM call
		await queue.put(('tc_active', {'step': current['step'], 'url': _cached_url(agent)}))

	async def on_new_step(browser_state, model_output, n_steps: int) -> None:
		s = _extract_step(model_output)
		if s:
			current['step'] = max(current['step'], s)  # monotonic — never jump backwards
		await queue.put(
			(
				'tc_active',
				{
					'step': current['step'],
					'url': getattr(browser_state, 'url', ''),
					'goal': getattr(model_output, 'next_goal', None),
					'actions': _summarize_actions(model_output),
				},
			)
		)

	async def on_step_end(agent) -> None:
		await queue.put(
			('tc_active', {'step': current['step'], 'url': await _live_url(agent), 'results': _summarize_results(agent)})
		)

	llm = build_llm()
	agent = Agent(
		task=task,
		llm=llm,
		browser_profile=build_browser_profile(device),
		sensitive_data=sensitive_data,
		use_vision=USE_AZURE,
		llm_timeout=300,
		step_timeout=600,
		register_new_step_callback=on_new_step,
	)
	# drop our local reference to the raw secret once it's handed to the agent
	secret = ''

	agent_task = asyncio.create_task(agent.run(max_steps=MAX_STEPS, on_step_start=on_step_start, on_step_end=on_step_end))

	# stream events; if the client disconnects (Stop), writing fails -> cancel the run
	while not (agent_task.done() and queue.empty()):
		try:
			event, item = await asyncio.wait_for(queue.get(), timeout=3.0)
		except asyncio.TimeoutError:
			event, item = 'ping', {'t': 1}
		try:
			await send(event, item)
		except (ConnectionResetError, ConnectionError, RuntimeError, asyncio.CancelledError):
			agent_task.cancel()
			break

	try:
		history = agent_task.result()
		shot = None
		try:
			shots = history.screenshots(n_last=1)
			shot = shots[0] if shots else None
		except Exception:
			shot = None
		await send('done', {'result': history.final_result(), 'screenshot': shot})
	except asyncio.CancelledError:
		try:
			await send('error_msg', {'error': 'run cancelled'})
		except Exception:
			pass
	except Exception as e:
		try:
			await send('error_msg', {'error': str(e)})
		except Exception:
			pass

	return resp


def main() -> None:
	app = web.Application()
	app.add_routes(
		[
			web.get('/', handle_index),
			web.post('/create', handle_create),
			web.post('/stream', handle_stream),
		]
	)
	print(f'browser-use web UI  ->  http://localhost:{PORT}  (backend: {BACKEND_LABEL})', flush=True)
	web.run_app(app, host='127.0.0.1', port=PORT, print=None)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
