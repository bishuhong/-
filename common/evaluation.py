"""Generation and structured-field metrics on held-out SFT test prompts."""
import json
from pathlib import Path
from common.data import *
from common.training import load_model,parser

def score(predictions):
    n=len(predictions)
    if not n: raise ValueError('No predictions')
    total={k:0 for k in ['json_parse_rate','schema_valid_rate','intent_set_accuracy','slots_exact_accuracy','next_action_accuracy',
                         'missing_slots_set_accuracy','evidence_set_accuracy','evidence_valid_rate','structured_exact_accuracy']}
    tp=fp=fn=0
    for row in predictions:
        gold=json.loads(row['expected'])
        try: out=json.loads(row['prediction'])
        except (json.JSONDecodeError,TypeError):
            fn+=len(gold['slots']); continue
        total['json_parse_rate']+=1
        if not isinstance(out,dict): fn+=len(gold['slots']); continue
        required={'intents','slots','missing_slots','next_action','reply','handoff_summary','evidence_ids'}
        schema=(set(out)==required and all(isinstance(out.get(k),list) and all(isinstance(x,str) for x in out[k]) for k in ['intents','missing_slots','evidence_ids'])
                and isinstance(out.get('slots'),dict) and isinstance(out.get('next_action'),str) and isinstance(out.get('reply'),str)
                and (out.get('handoff_summary') is None or isinstance(out.get('handoff_summary'),str)))
        total['schema_valid_rate']+=int(schema)
        matches=[]
        for field,metric in [('intents','intent_set_accuracy'),('missing_slots','missing_slots_set_accuracy'),('evidence_ids','evidence_set_accuracy')]:
            value=out.get(field)
            ok=isinstance(value,list) and all(isinstance(x,str) for x in value) and set(value)==set(gold[field])
            total[metric]+=int(ok); matches.append(ok)
        for field,metric in [('slots','slots_exact_accuracy'),('next_action','next_action_accuracy')]:
            ok=out.get(field)==gold[field]; total[metric]+=int(ok); matches.append(ok)
        total['structured_exact_accuracy']+=int(schema and all(matches))
        gs={(k,json.dumps(v,ensure_ascii=False,sort_keys=True)) for k,v in gold['slots'].items()}
        ps={(k,json.dumps(v,ensure_ascii=False,sort_keys=True)) for k,v in out.get('slots',{}).items()} if isinstance(out.get('slots'),dict) else set()
        tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps)
        refs=set()
        for m in row['prompt']:
            if m['role']=='user':
                try:
                    refs.update(x['id'] for x in json.loads(m['content']).get('context',{}).get('references',[]))
                except (ValueError,KeyError,TypeError): pass
        ev=out.get('evidence_ids')
        total['evidence_valid_rate']+=int(isinstance(ev,list) and all(isinstance(x,str) and x in refs for x in ev))
    result={k:v/n for k,v in total.items()}
    result.update(n=n,slot_precision=tp/(tp+fp) if tp+fp else 0,slot_recall=tp/(tp+fn) if tp+fn else 0)
    result['slot_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    result['note']='Automatic structural metrics only. Manually inspect facts, promises and handoff replies.'
    return result

def test(stage):
    a=parser(stage).parse_args(); model,tok,_=load_model(a,stage,False)
    import torch
    model.eval(); rows=read_jsonl(SFT_DATA/'test.jsonl')
    if a.limit: rows=rows[:a.limit]
    output=[]
    for i,row in enumerate(rows,1):
        prompt=row['messages'][:-1]
        inputs=tok(prompt_text(prompt),return_tensors='pt',add_special_tokens=False).to(model.device)
        if inputs['input_ids'].shape[1]+a.max_new_tokens>a.max_length:
            raise ValueError('Prompt plus generation budget exceeds --max-length; increase it.')
        with torch.inference_mode():
            result=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                                  eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        generated=result[0,inputs['input_ids'].shape[1]:]
        output.append({'index':i,'prompt':prompt,'expected':row['messages'][-1]['content'],
                       'prediction':tok.decode(generated,skip_special_tokens=True).strip()})
        print(f'{stage}: {i}/{len(rows)}',flush=True)
    dest=ROOT/'reports'
    write_jsonl(dest/f'{stage.lower()}_test_predictions.jsonl',output)
    metrics=score(output); metrics['smoke_subset']=bool(a.limit)
    write_json(dest/f'{stage.lower()}_test_metrics.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
