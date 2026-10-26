"""Standard-library data checks and one consistent ChatML format."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_DATA = ROOT / 'data/ecommerce_sft_v1'
DPO_DATA = ROOT / 'DPO/data'
SPLITS = ('train', 'validation', 'test')
# Base checkpoint has no instruction-tuned behavior. Explicitly teach ChatML.
CHAT_TEMPLATE = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def read_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8-sig') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{i}: {e}') from e
    if not rows: raise ValueError(f'Empty dataset: {path}')
    return rows

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows),encoding='utf-8')

def prompt_text(messages):
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)+'<|im_start|>assistant\n'

def completion_text(text):
    return text+'<|im_end|>\n'

def check_messages(messages):
    if len(messages)<2 or messages[-1]['role']!='assistant':
        raise ValueError('Expected conversation ending in an assistant response')
    for m in messages:
        if m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) or not m['content'].strip():
            raise ValueError('Invalid message')
        if any(t in m['content'] for t in ('<|im_start|>','<|im_end|>')):
            raise ValueError('Raw ChatML special token in content')
    json.loads(messages[-1]['content'])


def audit_sft():
    seen={}; counts={}; annotated={s:[] for s in SPLITS}
    for a in read_jsonl(SFT_DATA/'annotations.jsonl'): annotated[a['split']].append(a)
    families={}
    for split in SPLITS:
        rows=read_jsonl(SFT_DATA/f'{split}.jsonl'); counts[split]=len(rows)
        if len(annotated[split])!=len(rows): raise ValueError('Annotation count mismatch')
        for row in rows:
            check_messages(row['messages'])
            key=json.dumps(row['messages'][:-1],ensure_ascii=False,sort_keys=True)
            if key in seen: raise ValueError(f'Duplicate prompt: {seen[key]}, {split}')
            seen[key]=split
        for a in annotated[split]:
            fid=a['family_id']
            if fid in families and families[fid]!=split: raise ValueError('Scenario leakage')
            families[fid]=split
            expected=json.loads(rows[a['jsonl_line']-1]['messages'][-1]['content'])
            if expected!=a['expected']: raise ValueError('Annotation mismatch')
    result={'counts':counts,'families':len(families),'train_families':sum(v=='train' for v in families.values()),
            'sha256':{s:hashlib.sha256((SFT_DATA/f'{s}.jsonl').read_bytes()).hexdigest() for s in SPLITS},
            'limitations':['180 synthetic examples, only 36 scenario families; not production evidence.',
                           'DPO requires preference pairs; existing data is SFT only.']}
    return result


def dpo_rows(split, allow_unreviewed=False):
    result=[]; seen=set()
    for row in read_jsonl(DPO_DATA/f'{split}.jsonl'):
        if row.get('reviewed') is not True and not allow_unreviewed:
            raise ValueError('Unreviewed DPO pairs. Review and set reviewed=true, or explicitly use --allow-unreviewed for a synthetic pipeline experiment.')
        if not isinstance(row.get('prompt'),list) or not row['prompt'] or row['prompt'][-1]['role']!='user':
            raise ValueError('DPO prompt must be messages ending in user')
        check_messages(row['prompt']+[{'role':'assistant','content':row['chosen']}])
        check_messages(row['prompt']+[{'role':'assistant','content':row['rejected']}])
        if row['chosen']==row['rejected']: raise ValueError('Identical preference pair')
        key=prompt_text(row['prompt'])
        if key in seen: raise ValueError('Duplicate DPO prompt')
        seen.add(key)
        result.append({'prompt':key,'chosen':completion_text(row['chosen']),'rejected':completion_text(row['rejected'])})
    return result
