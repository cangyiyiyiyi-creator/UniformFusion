#!/usr/bin/env python3
import argparse,json,time,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets as D
from tools.profile_project_checkpoint import load_model,autocast_context,synchronize
from main_finetune import set_seed

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--method-label',required=True);p.add_argument('--list',default='annotations/DvXray_test.txt');p.add_argument('--classes-file',default='annotations/classes.txt');p.add_argument('--batch-size',type=int,required=True);p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=100);p.add_argument('--num-workers',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--output-json',required=True);p.add_argument('--seed',type=int,default=20260907)
a=p.parse_args();out=Path(a.output_json)
if out.exists():raise FileExistsError(out)
set_seed(a.seed,deterministic=True);device=torch.device(a.device);model,ckpt,args=load_model(Path(a.checkpoint),device)
names=D._read_class_names(a.classes_file,int(getattr(args,'num_classes',15)))
ds=D.DualViewTxtDataset(a.list,int(getattr(args,'input_size',224)),len(names),train=False,class_names=names,view_mode=str(getattr(args,'view_mode','paired')))
loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,drop_last=False,persistent_workers=a.num_workers>0)
it=iter(loader)
def step():
 global it
 try:batch=next(it)
 except StopIteration:it=iter(loader);batch=next(it)
 views=batch[0]
 if not isinstance(views,(list,tuple)) or len(views)!=2:
  raise ValueError(f'expected ((view_a, view_b), target), got {type(batch)!r}')
 xa,xb=views[0].to(device,non_blocking=True),views[1].to(device,non_blocking=True)
 with torch.inference_mode(),autocast_context(device):model(xa,xb)
for _ in range(a.warmup):step()
synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
for _ in range(a.repeats):step()
synchronize(device);elapsed=time.perf_counter()-start
result={'method':a.method_label,'checkpoint':a.checkpoint,'scope':'dataloader_h2d_model','device':str(device),'precision':'amp_fp16','batch_size':a.batch_size,'warmup':a.warmup,'repeats':a.repeats,'latency_ms_per_batch':1000*elapsed/a.repeats,'latency_ms_per_sample':1000*elapsed/a.repeats/a.batch_size,'throughput_samples_per_s':a.batch_size*a.repeats/elapsed,'peak_memory_MiB':torch.cuda.max_memory_allocated(device)/2**20}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print('END_TO_END_PROFILE_OK',a.method_label,a.batch_size,result['latency_ms_per_batch'])
