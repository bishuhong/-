"""Single-GPU QLoRA, with explicit final-answer masking and no truncation."""
import argparse
import json
import math
import os
from pathlib import Path
from common.data import *

def parser(stage):
    p=argparse.ArgumentParser()
    p.add_argument('--model',default=str(ROOT/'models/Qwen3-8B-Base'))
    p.add_argument('--output',default=str(ROOT/f'outputs/{stage.lower()}'))
    p.add_argument('--sft-adapter',default=str(ROOT/'outputs/sft'))
    p.add_argument('--adapter',default=None,help='Adapter used for validation/test; defaults to stage output')
    p.add_argument('--max-length',type=int,default=2048)
    p.add_argument('--epochs',type=float,default=2 if stage=='SFT' else 1)
    p.add_argument('--lr',type=float,default=1e-4 if stage=='SFT' else 5e-6)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--beta',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--resume',default=None)
    p.add_argument('--allow-unreviewed',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--limit',type=int,default=0,help='0 means full split; nonzero is a smoke test only')
    return p

def hardware():
    import torch
    if int(os.environ.get('WORLD_SIZE','1'))!=1: raise RuntimeError('This configuration is for one GPU; launch with python, not torchrun.')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable. Install CUDA-enabled PyTorch on the cloud GPU.')
    bf16=torch.cuda.is_bf16_supported()
    return torch, torch.bfloat16 if bf16 else torch.float16, bf16

def tokenizer_for(path):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path)
    tok.chat_template=CHAT_TEMPLATE
    # end-of-turn is the supervised stop token, distinct from <|endoftext|>.
    tok.eos_token='<|im_end|>'
    tok.pad_token='<|endoftext|>'
    if tok.convert_tokens_to_ids('<|im_end|>')==tok.unk_token_id: raise ValueError('Tokenizer lacks ChatML tokens')
    tok.padding_side='right'
    return tok

def load_model(a, stage, training):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    torch,dtype,bf16=hardware()
    tok=tokenizer_for(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map={'':0},
        attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    model.config.use_cache=not training
    model.config.eos_token_id=tok.eos_token_id
    model.config.pad_token_id=tok.pad_token_id
    model.generation_config.eos_token_id=tok.eos_token_id
    model.generation_config.pad_token_id=tok.pad_token_id
    if training: model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    if stage=='SFT' and training:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    elif stage=='DPO' and training:
        if not (Path(a.sft_adapter)/'adapter_config.json').exists(): raise FileNotFoundError('Run SFT first; --sft-adapter must point to its adapter output.')
        model=PeftModel.from_pretrained(model,a.sft_adapter,adapter_name='default',is_trainable=True)
        model.load_adapter(a.sft_adapter,adapter_name='reference',is_trainable=False)
        model.set_adapter('default')
        for name,param in model.named_parameters():
            if '.reference.' in name: param.requires_grad=False
    else:
        path=a.adapter or a.output
        model=PeftModel.from_pretrained(model,path,is_trainable=False)
    return model,tok,bf16

def encode_sft(tok, row, max_length):
    messages=row['messages']; check_messages(messages)
    prompt=tok.encode(prompt_text(messages[:-1]),add_special_tokens=False)
    target=tok.encode(completion_text(messages[-1]['content']),add_special_tokens=False)
    ids=prompt+target
    if len(ids)>max_length:
        raise ValueError(f'{len(ids)} tokens > {max_length}. Increase --max-length; silent truncation is forbidden.')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prompt)+target}

def sft_dataset(tok, split, max_length):
    from datasets import Dataset
    data=[]
    for i,row in enumerate(read_jsonl(SFT_DATA/f'{split}.jsonl'),1):
        try: data.append(encode_sft(tok,row,max_length))
        except ValueError as e: raise ValueError(f'{split}:{i}: {e}') from e
    return Dataset.from_list(data)

class CausalCollator:
    def __init__(self, pad_id): self.pad_id=pad_id
    def __call__(self, features):
        import torch
        n=max(len(x['input_ids']) for x in features)
        return {k:torch.tensor([x[k]+[(-100 if k=='labels' else self.pad_id if k=='input_ids' else 0)]*(n-len(x[k])) for x in features],dtype=torch.long)
                for k in ('input_ids','attention_mask','labels')}

def base_args(a,bf16):
    return dict(output_dir=a.output,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,num_train_epochs=a.epochs,learning_rate=a.lr,
        bf16=bf16,fp16=not bf16,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},optim='paged_adamw_8bit',
        warmup_ratio=0.1,lr_scheduler_type='cosine',max_grad_norm=1.0,logging_steps=1,
        eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,
        metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',seed=a.seed,
        data_seed=a.seed,dataloader_num_workers=0)

def trainer_for(a,stage,training):
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(a.seed)
    audit_sft()
    if stage=='DPO':
        # Reject unreviewed data BEFORE allocating VRAM.
        train=dpo_rows('train',a.allow_unreviewed); val=dpo_rows('validation',a.allow_unreviewed)
    model,tok,bf16=load_model(a,stage,training)
    if stage=='SFT':
        trainer=Trainer(model=model,args=TrainingArguments(**base_args(a,bf16)),
            train_dataset=sft_dataset(tok,'train',a.max_length),
            eval_dataset=sft_dataset(tok,'validation',a.max_length),
            data_collator=CausalCollator(tok.pad_token_id),processing_class=tok)
    else:
        from datasets import Dataset
        from trl import DPOConfig,DPOTrainer
        for split,rows in [('train',train),('validation',val)]:
            for i,row in enumerate(rows,1):
                for key in ('chosen','rejected'):
                    # DPOTrainer adds EOS; allow a spare token and refuse truncation.
                    n=len(tok.encode(row['prompt'],add_special_tokens=False))+len(tok.encode(row[key],add_special_tokens=False))+1
                    if n>a.max_length: raise ValueError(f'DPO {split}:{i}: {n} > {a.max_length}')
        cfg=DPOConfig(**base_args(a,bf16),beta=a.beta,max_length=a.max_length,max_prompt_length=a.max_length,
                      model_adapter_name='default',ref_adapter_name='reference',precompute_ref_log_probs=False)
        if not training:
            # Inference adapter is the DPO policy; reference always stays original SFT.
            model.load_adapter(a.sft_adapter,adapter_name='reference',is_trainable=False)
        trainer=DPOTrainer(model=model,ref_model=None,args=cfg,processing_class=tok,
                           train_dataset=Dataset.from_list(train),eval_dataset=Dataset.from_list(val))
    return trainer,tok

def train(stage):
    a=parser(stage).parse_args()
    trainer,tok=trainer_for(a,stage,True)
    result=trainer.train(resume_from_checkpoint=a.resume)
    trainer.model.save_pretrained(a.output,selected_adapters=['default'])
    tok.save_pretrained(a.output)
    trainer.save_state()
    trainer.log_metrics('train',result.metrics); trainer.save_metrics('train',result.metrics)
    trainer.save_metrics('validation',trainer.evaluate())
    write_json(Path(a.output)/'run_config.json',vars(a))

def validate(stage):
    a=parser(stage).parse_args()
    trainer,_=trainer_for(a,stage,False)
    metrics=trainer.evaluate()
    if stage=='SFT': metrics['perplexity']=math.exp(min(metrics['eval_loss'],50))
    # Output reports outside checkpoints to avoid changing a model by evaluating it.
    write_json(ROOT/f'reports/{stage.lower()}_validation.json',metrics)
    print(json.dumps(metrics,indent=2))
