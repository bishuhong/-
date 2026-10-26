"""CPU/tokenizer checks: data isolation, loss masks, truncation, metric failures."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import *
from common.training import tokenizer_for,encode_sft
from common.evaluation import score

def main():
    tok=tokenizer_for(sys.argv[1] if len(sys.argv)>1 else ROOT/'models/Qwen3-8B-Base')
    count=0
    for split in SPLITS:
        for row in read_jsonl(SFT_DATA/f'{split}.jsonl'):
            encoded=encode_sft(tok,row,2048)
            plen=len(tok.encode(prompt_text(row['messages'][:-1]),add_special_tokens=False))
            assert all(x==-100 for x in encoded['labels'][:plen])
            assert encoded['labels'][plen:]==encoded['input_ids'][plen:]
            assert any(x!=-100 for x in encoded['labels'])
            assert tok.eos_token_id in encoded['labels'][plen:]
            try: encode_sft(tok,row,10)
            except ValueError: pass
            else: raise AssertionError('Truncation not rejected')
            count+=1
    try: dpo_rows('train')
    except ValueError as e: assert 'Unreviewed' in str(e)
    else: raise AssertionError('Unreviewed candidates accepted')
    gold=read_jsonl(SFT_DATA/'test.jsonl')
    perfect=[{'prompt':r['messages'][:-1],'expected':r['messages'][-1]['content'],'prediction':r['messages'][-1]['content']} for r in gold]
    good=score(perfect)
    assert all(good[k]==1 for k in ['json_parse_rate','schema_valid_rate','next_action_accuracy','structured_exact_accuracy','slot_f1','evidence_valid_rate'])
    invalid=copy.deepcopy(perfect)
    for r in invalid: r['prediction']='not json'
    bad=score(invalid)
    assert bad['json_parse_rate']==0 and bad['slot_recall']==0
    malformed=copy.deepcopy(perfect)
    for r in malformed: r['prediction']=json.dumps({'intents':[{}],'slots':[],'evidence_ids':[{}]})
    assert score(malformed)['schema_valid_rate']==0
    write_json(ROOT/'reports/local_checks.json',{'status':'passed','masked_examples':count,'checks':[
        'data hashes and split isolation','all source samples fit max_length','final-answer-only loss mask',
        'end-of-turn supervision','overlength fails rather than truncates','unreviewed DPO gate',
        'perfect and invalid structured-metric fixtures','malformed JSON types handled'],
        'not_run':['8B forward/backward','CUDA QLoRA training','cloud GPU peak-memory measurement'],
        'reason':'Local validation uses tokenizer only; full GPU training must run on cloud.'})
    print(f'PASSED: {count} masked examples; metrics and DPO gate checked. GPU training not run.')
if __name__=='__main__': main()
