import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import ROOT,dpo_rows,write_json
from common.training import parser,trainer_for

def main():
    from datasets import Dataset
    a=parser('DPO').parse_args()
    rows=dpo_rows('test',a.allow_unreviewed)
    trainer,_=trainer_for(a,'DPO',False)
    metrics=trainer.evaluate(eval_dataset=Dataset.from_list(rows),metric_key_prefix='test')
    write_json(ROOT/'reports/dpo_preference_test.json',metrics)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
