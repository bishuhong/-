"""Fail early on bad data, lengths, incomplete checkpoints or GPU environment."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import *
from common.training import tokenizer_for

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-only',action='store_true')
    p.add_argument('--model',default=str(ROOT/'models/Qwen3-8B-Base')); p.add_argument('--max-length',type=int,default=2048)
    a=p.parse_args(); report=audit_sft(); tok=tokenizer_for(a.model)
    lengths={}; prompts={}
    for split in SPLITS:
        sizes=[]; keys=set()
        for row in read_jsonl(SFT_DATA/f'{split}.jsonl'):
            prompt=prompt_text(row['messages'][:-1]); answer=completion_text(row['messages'][-1]['content'])
            # Check that the explicit template exactly matches inference tokenization.
            assert tok.apply_chat_template(row['messages'][:-1],tokenize=False,add_generation_prompt=True)==prompt
            n=len(tok.encode(prompt,add_special_tokens=False))+len(tok.encode(answer,add_special_tokens=False))
            if n>a.max_length: raise ValueError(f'{split}: {n} > {a.max_length}')
            sizes.append(n); keys.add(prompt)
        lengths[split]={'max':max(sizes),'mean':sum(sizes)/len(sizes)}; prompts[split]=keys
    if DPO_DATA.exists():
        dpo_keys={}; report['dpo']={}
        for split in SPLITS:
            rows=dpo_rows(split,allow_unreviewed=True); dpo_keys[split]={r['prompt'] for r in rows}
            if dpo_keys[split]!=prompts[split]: raise ValueError('DPO must preserve original split membership')
            for row in rows:
                for field in ('chosen','rejected'):
                    n=len(tok.encode(row['prompt'],add_special_tokens=False))+len(tok.encode(row[field],add_special_tokens=False))+1
                    if n>a.max_length: raise ValueError('DPO would truncate')
            report['dpo'][split]={'count':len(rows),'reviewed':sum(x.get('reviewed') is True for x in read_jsonl(DPO_DATA/f'{split}.jsonl'))}
        for i,s in enumerate(SPLITS):
            for t in SPLITS[i+1:]:
                if dpo_keys[s]&dpo_keys[t]: raise ValueError('DPO leakage')
    report['token_lengths']=lengths
    if not a.data_only:
        import torch
        import bitsandbytes
        import trl
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if torch.cuda.device_count()!=1: raise RuntimeError('Expose one GPU with CUDA_VISIBLE_DEVICES=0')
        x=torch.randn(128,128,device='cuda'); y=x@x; torch.cuda.synchronize()
        report['gpu']={'name':torch.cuda.get_device_name(0),'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
                       'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,'bf16':torch.cuda.is_bf16_supported()}
        model=Path(a.model); index=model/'model.safetensors.index.json'
        names=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        if any(not (model/n).exists() for n in names): raise RuntimeError('Model weights incomplete. Run download_model.py first.')
        report['packages']={n:importlib.metadata.version(n) for n in ['transformers','trl','peft','accelerate','datasets','bitsandbytes']}
    write_json(ROOT/'reports/preflight_data.json' if a.data_only else ROOT/'reports/preflight_gpu.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
