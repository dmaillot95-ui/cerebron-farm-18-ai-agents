import os,json,time
from pathlib import Path
try:
 from hf_gradio import GradioClient
except Exception as e:
 GradioClient=None
role=os.environ.get('ROLE','UNKNOWN')
model=os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct')
focus=os.environ.get('FOCUS','AI agent systems')
out=Path('out'); out.mkdir(exist_ok=True)
prompt=f'''You are role {role} in CEREBRON Farm 18 AI Agents. Focus: {focus}.
Rules: CLAIM<=EVIDENCE; AGENT COUNT != INTELLIGENCE; same model/data are not independent evidence; tool call is not success until verified; memory != learning; simulation != test. Return concise findings, failure modes, verification steps, measurable tests, and unknowns. Do not claim executions you did not perform.'''
res={'role':role,'model':model,'focus':focus,'inference_success':False,'epistemic_status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT'}
try:
 if GradioClient is None: raise RuntimeError('hf_gradio unavailable')
 c=GradioClient(model)
 ans=c.predict(message=prompt,api_name='/chat')
 res['inference_success']=True; res['result']=str(ans)
except Exception as e:
 res['error']=repr(e)
(out/f'{role}.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps(res,ensure_ascii=False))
