import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lock=ROOT/'models/model_lock.json'
manifest=json.loads(lock.read_text()) if lock.exists() else {}
p=ROOT/'models/Qwen3-8B-Base'
complete=list(p.glob('*.safetensors'))
partial=list(p.glob('*.safetensors.part'))
received=sum(f.stat().st_size for f in complete+partial)
total=manifest.get('weights_bytes',0)
print(json.dumps({'status':manifest.get('status','not_started'),'model':manifest.get('model_id'),
                  'revision':manifest.get('revision'),'received_bytes':received,'total_bytes':total,
                  'progress_percent':round(received/total*100,2) if total else 0,
                  'verified_shards':len(complete),'partial_shards':len(partial)},indent=2))
