import argparse, csv, json, statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
    rows=[]
    for f in sorted(root.glob('resnet50/seed_*/repeat_*/*/test_metrics.json')):
        d=json.loads(f.read_text()); method=f.parent.name; seed=int(f.parents[2].name.split('_')[-1])
        v=json.loads((f.parent/'val_metrics.json').read_text())
        rows.append({'method':method,'seed':seed,'val_mAP':v['stats']['mAP'],'test_mAP':d['stats']['mAP'],'checkpoint':str(f.parent/'checkpoint_best.pth')})
    if len(rows)!=9: raise RuntimeError(f'expected 9 completed tests, found {len(rows)}')
    with (root/'first_batch_9runs_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        x=[r for r in rows if r['method']==method]; vals=[float(r['test_mAP']) for r in x]
        summary.append({'method':method,'n':len(x),'test_mAP_mean':statistics.mean(vals),'test_mAP_std_sample':statistics.stdev(vals)})
    with (root/'first_batch_9runs_summary.csv').open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    print('REVIEWER_FIRST_BATCH_SUMMARY_OK runs=9')
if __name__=='__main__': main()
