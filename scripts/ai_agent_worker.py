#!/usr/bin/env python3
import os,json,pathlib,subprocess,hashlib
PREFERRED=['/generate','/chat','/predict','/respond','/infer','/run']
def run(cmd,timeout=240): return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
def payload_for(spec,prompt):
 p={}; set_prompt=False
 for x in spec.get('parameters',[]):
  n=x.get('name',''); l=n.lower(); req=bool(x.get('required',False)); default=x.get('default'); typ=(x.get('type') or {}).get('type')
  if l in {'message','prompt','text','query','input','instruction','user_message'}: p[n]=prompt; set_prompt=True
  elif l in {'chat_history','history','messages'}: p[n]=[]
  elif l in {'max_new_tokens','max_tokens','maximum_new_tokens'}: p[n]=900
  elif l=='temperature': p[n]=0.1
  elif l=='top_p': p[n]=0.9
  elif l=='top_k': p[n]=40
  elif l in {'system','system_prompt'}: p[n]='Evidence-grounded AI-agent analysis. CLAIM<=EVIDENCE. Never invent tool calls or executions.'
  elif req and default is None:
   if typ=='string' and not set_prompt: p[n]=prompt; set_prompt=True
   else: return None
 return p if set_prompt else None
def extract(raw):
 raw=raw.strip()
 try:
  o=json.loads(raw)
  if isinstance(o,dict):
   for k in ('Response','response','text','output','message'):
    if isinstance(o.get(k),str): return o[k].strip()
 except: pass
 return raw
def invoke(space,prompt):
 info=run(['hf-gradio','info',space],120)
 if info.returncode!=0: return False,'',{'stage':'info','error':(info.stderr or info.stdout)[-1200:]}
 try: api=json.loads(info.stdout)
 except Exception as e: return False,'',{'stage':'decode','error':repr(e)}
 eps=list(api.items()); eps.sort(key=lambda kv:(PREFERRED.index(kv[0]) if kv[0] in PREFERRED else 99,kv[0])); errs=[]
 for ep,spec in eps:
  payload=payload_for(spec,prompt)
  if payload is None: continue
  pred=run(['hf-gradio','predict',space,ep,json.dumps(payload,ensure_ascii=False)],240)
  if pred.returncode==0 and (pred.stdout or '').strip():
   text=extract(pred.stdout)
   if text: return True,text,{'endpoint':ep,'sha256':hashlib.sha256(text.encode()).hexdigest()}
  errs.append((pred.stderr or pred.stdout)[-700:])
 return False,'',{'stage':'predict','error':' | '.join(errs[-3:]) or 'No compatible endpoint'}
role=os.environ.get('ROLE','UNKNOWN'); model=os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct'); focus=os.environ.get('FOCUS','AI agent systems')
prompt=f'''You are role {role} in CEREBRON Farm 18 AI Agents. Focus: {focus}. Rules: CLAIM<=EVIDENCE; AGENT COUNT != INTELLIGENCE; same model/data are not independent evidence; tool call is not success until verified; memory != learning; simulation != test. Return concise findings, failure modes, verification steps, measurable tests, and unknowns. Do not claim executions you did not perform.'''
ok,text,meta=invoke(model,prompt)
res={'role':role,'model':model,'focus':focus,'inference_success':bool(ok),'status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT' if ok else 'EXTERNAL_INFERENCE_FAILED','epistemic_status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT' if ok else 'EXTERNAL_INFERENCE_FAILED','result':text if ok else None,'error':None if ok else meta.get('error'),'meta':meta}
out=pathlib.Path('out'); out.mkdir(exist_ok=True); (out/f'{role}.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)); print(json.dumps({'role':role,'inference_success':bool(ok),'model':model}))