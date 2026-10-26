"""Download official snapshot; pin commit and verify every weight shard exists."""
import argparse
import json
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
ROOT=Path(__file__).resolve().parents[1]

def ranged_download(repo, revision, file, dest, endpoint):
    path=dest/file.rfilename
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix(path.suffix+'.part')
    sha=file.lfs.sha256 if file.lfs else None
    if path.exists() and path.stat().st_size==file.size:
        if sha:
            with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=sha: raise RuntimeError(f'Checksum mismatch: {path}')
        return
    url=f'{endpoint}/{repo}/resolve/{revision}/{file.rfilename}'
    offset=part.stat().st_size if part.exists() else 0
    if offset>file.size: raise RuntimeError(f'Oversized partial file: {part}')
    while offset<file.size:
        end=min(offset+8*1024*1024,file.size)-1
        for attempt in range(5):
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(15,60)) as response:
                    response.raise_for_status()
                    expected=f'bytes {offset}-{end}/{file.size}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise RuntimeError('Server did not honor exact byte range')
                    # Buffer one range so an interrupted request cannot append partial data.
                    chunk=response.content
                    if len(chunk)!=end-offset+1: raise RuntimeError('Incomplete HTTP range')
                with part.open('ab') as f: f.write(chunk)
                offset=end+1
                if offset%(64*1024*1024)==0 or offset==file.size:
                    print(f'{file.rfilename}: {offset}/{file.size} bytes ({offset/file.size:.1%})',flush=True)
                break
            except (requests.RequestException,RuntimeError):
                if attempt==4: raise
                time.sleep(min(2**attempt,16))
    if sha:
        with part.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=sha: raise RuntimeError(f'Checksum mismatch: {part}; do not use this file')
    part.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen3-8B-Base')
    p.add_argument('--revision',default=None)
    p.add_argument('--output',type=Path,default=ROOT/'models/Qwen3-8B-Base')
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--http-chunks',action='store_true',help='Use resumable 8 MiB ranges for proxies that stall large downloads')
    a=p.parse_args()
    lock=ROOT/'models/model_lock.json'
    revision=a.revision
    if revision is None and lock.exists():
        saved=json.loads(lock.read_text(encoding='utf-8'))
        if saved['model_id']==a.model: revision=saved['revision']
    info=HfApi().model_info(a.model,revision=revision,files_metadata=True)
    manifest={'model_id':a.model,'revision':info.sha,'endpoint':os.environ.get('HF_ENDPOINT','https://huggingface.co'),
              'weights_bytes':sum(f.size or 0 for f in info.siblings if f.rfilename.endswith('.safetensors')),
              'status':'metadata_only' if a.metadata_only else 'downloading'}
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    patterns=['*.json','*.jinja','*.txt','*.model','README.md','LICENSE']
    try:
        snapshot_download(a.model,revision=info.sha,local_dir=a.output,
                          allow_patterns=patterns if a.metadata_only or a.http_chunks else ['*.safetensors']+patterns,max_workers=4)
        if a.http_chunks and not a.metadata_only:
            weights=[f for f in info.siblings if f.rfilename.endswith('.safetensors')]
            endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co')
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures=[pool.submit(ranged_download,a.model,info.sha,f,a.output,endpoint) for f in weights]
                for future in futures: future.result()
    except Exception as error:
        manifest['status']='failed'; manifest['error_type']=type(error).__name__
        lock.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        raise
    if not a.metadata_only:
        index=a.output/'model.safetensors.index.json'
        weights=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        remote={f.rfilename:f.size for f in info.siblings}
        for name in weights:
            file=a.output/name
            if not file.exists() or file.stat().st_size != remote[name]: raise RuntimeError(f'Incomplete shard: {name}')
        manifest['status']='complete'; manifest['local_path']=str(a.output.resolve())
        lock.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Download verified.' if not a.metadata_only else 'Metadata only; weights NOT downloaded.',flush=True)
if __name__=='__main__': main()
