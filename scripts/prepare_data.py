"""Audit original data; make explicitly unreviewed synthetic DPO candidates."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--build-dpo',action='store_true'); a=p.parse_args()
    report=audit_sft()
    write_json(ROOT/'reports/data_audit.json',report)
    if a.build_dpo:
        # Preserve existing reviewed work. Never overwrite a file of preferences.
        for split in SPLITS:
            dest=DPO_DATA/f'{split}.jsonl'
            if dest.exists(): raise FileExistsError(f'{dest} already exists; preserving it')
        anns={(a['split'],a['jsonl_line']):a for a in read_jsonl(SFT_DATA/'annotations.jsonl')}
        for split in SPLITS:
            out=[]
            for i,row in enumerate(read_jsonl(SFT_DATA/f'{split}.jsonl'),1):
                ann=anns[split,i]; chosen=row['messages'][-1]['content']; bad=copy.deepcopy(json.loads(chosen))
                # Clearly wrong, schema-preserving action: requesting a password is
                # explicitly prohibited in every source system message.
                bad['next_action']='clarify'
                bad['missing_slots']=['account_password']
                bad['reply']='请先提供您的账号登录密码，我才能继续处理。'
                out.append({'id':ann['sample_id'],'family_id':ann['family_id'],'split':split,
                            'prompt':row['messages'][:-1],'chosen':chosen,
                            'rejected':json.dumps(bad,ensure_ascii=False),
                            'source':'synthetic_rule_corruption_v1','reviewed':False,
                            'reason':'Rejected solicits an account password contrary to source rules. Easy synthetic negative; not human preference evidence.'})
            write_jsonl(DPO_DATA/f'{split}.jsonl',out)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
