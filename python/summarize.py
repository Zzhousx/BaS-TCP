import argparse,csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import stats

NAMES=dict(bas='BaS-TCP',greedy='Greedy-TCP',original='Original Order',strap='STRaP',random='Random Order',
 art='ART',cluster='Cluster-TCP',spectre='SPECTRE*',rl='RL-static*',least='LEAST-static*',ga='GA-TCP',mo='MO-TCP',
 raw='Raw features',pca='PCA',umap='UMAP',vae='VAE',lle='LLE',isa='ISA*')
ROOT=Path(__file__).resolve().parents[1]

def score(order,y,c):
    n=len(y);m=y.sum()
    if np.all(c==1):return 1-(np.flatnonzero(y[order])+1).sum()/(n*m)+1/(2*n)
    return np.trapezoid(np.cumsum(y[order]),np.cumsum(c[order]))/(c.sum()*m)

def records(directory):
    rows=[]
    for p in sorted(directory.glob('*.mat')):
        r=loadmat(p,simplify_cells=True)['R'];d=int(r['dataset']);D=loadmat(ROOT/f'data/Dataset{d}.mat',squeeze_me=True)
        order=np.asarray(r['order']).ravel().astype(int)-1;y=D['y'].astype(bool);c=D['costs']
        assert np.array_equal(np.sort(order),np.arange(len(y))),p
        s=score(order,y,c);assert abs(s-r['score'])<1e-9,(p,s,r['score'])
        rows.append(dict(dataset=d,view=str(r.get('view','full')),method=str(r['method']),seed=int(r['seed']),score=s,
            seconds=float(r.get('seconds',0))))
    return rows

def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['results','outputs'],default='results');args=ap.parse_args()
    source=ROOT/args.source;out=ROOT/'outputs/summary';out.mkdir(parents=True,exist_ok=True)
    for category in ['effectiveness','components']:
        raw=records(source/category);write(out/f'{category}_scores.csv',raw);groups={}
        for r in raw:groups.setdefault((r['dataset'],r['view'],r['method']),[]).append(r)
        summary=[]
        for (d,v,m),rows in sorted(groups.items()):
            x=np.array([r['score'] for r in rows]);n=len(x)
            if n>1:assert sorted(r['seed'] for r in rows)==list(range(1,31))
            summary.append(dict(dataset=d,view=v,method=NAMES.get(m,m),runs=n,mean=x.mean(),sd=x.std(ddof=1) if n>1 else 0))
        write(out/f'{category}.csv',summary)
        if category=='effectiveness':
            tests=[]
            for (d,v,m),rows in sorted(groups.items()):
                if len(rows)==1:continue
                reference=groups[d,v,'bas'][0]['score'];x=np.array([r['score'] for r in rows]);se=x.std(ddof=1)/np.sqrt(len(x));critical=stats.t.ppf(.975,len(x)-1)
                t=(reference-x.mean())/se if se else np.nan;p=float(2*stats.t.sf(abs(t),len(x)-1)) if se else 1.
                tests.append(dict(dataset=d,view=v,method=NAMES[m],t=t,p=p,holm_p=None,difference=reference-x.mean(),
                    ci_low=reference-x.mean()-critical*se,ci_high=reference-x.mean()+critical*se))
            for d,v in {(r['dataset'],r['view']) for r in tests}:
                family=sorted([r for r in tests if (r['dataset'],r['view'])==(d,v)],key=lambda r:r['p']);previous=0
                for i,r in enumerate(family):previous=max(previous,(len(family)-i)*r['p']);r['holm_p']=min(1.,previous)
            write(out/'statistics.csv',tests)
    timing=source/'timing.csv' if args.source=='results' else source/'timing/timing.csv'
    if timing.exists():
        rows=list(csv.DictReader(timing.open()));result=[]
        aliases=[('NmSTrainSeconds','Learning'),('NmSSearchSeconds','Search'),('NmSTotalSeconds','Total'),('GreedySeconds','Greedy')]
        for d in [1,2,3]:
            r=[r for r in rows if str(r['Dataset']) in [str(d),f'Dataset{d}']];assert len(r)==15
            for old,new in aliases:
                x=np.array([float(a.get(old,a.get(new))) for a in r]);result.append(dict(dataset=d,phase=new,mean=x.mean(),sd=x.std(ddof=1)))
        write(out/'timing.csv',result)
    sparsity=[]
    for p in sorted((source/'sparsity').glob('D*.mat')):
        r=loadmat(p,simplify_cells=True)['R'];d=int(r['dataset']);D=loadmat(ROOT/f'data/Dataset{d}.mat',squeeze_me=True)
        o=np.asarray(r['order']).ravel().astype(int)-1;matched=np.asarray(r.get('matched_order',[])).ravel().astype(int)-1
        native=score(o,D['y'].astype(bool),D['costs']);assert abs(native-r['score'])<1e-9
        sparsity.append(dict(dataset=d,budget=int(np.ceil(D['X'].shape[1]/2)),native_support=int((np.linalg.norm(r['W'],axis=1)>1e-8).sum()),
            native_score=native,refit_score=score(matched,D['y'].astype(bool),D['costs']),rho=r['rho']))
    write(out/'sparsity.csv',sparsity)
    costs=[]
    for d,target,factor in [(1,2584,10),(2,261,1)]:
        D=loadmat(ROOT/f'data/Dataset{d}.mat',squeeze_me=True);y=D['y'].astype(bool);c=D['costs']
        groups={}
        for p in sorted((source/'effectiveness').glob(f'D{d}_full_*.mat')):
            r=loadmat(p,simplify_cells=True)['R'];groups.setdefault(str(r['method']),[]).append(r)
        for m,runs in groups.items():
            executions=[];overheads=[]
            for r in runs:
                ix=np.asarray(r['order']).ravel().astype(int)-1;cut=np.searchsorted(np.cumsum(y[ix]),target)
                executions.append(c[ix[:cut+1]].sum()*factor);overheads.append(r['seconds'])
            overhead=float(np.mean(overheads))
            if m in ['bas','greedy'] and timing.exists():
                phase='Total' if m=='bas' else 'Greedy';overhead=next(r['mean'] for r in result if r['dataset']==d and r['phase']==phase)
            execution=float(np.mean(executions));costs.append(dict(dataset=d,method=NAMES[m],target=target,prioritization_seconds=overhead,
                execution_seconds=execution,total_hours=(overhead+execution)/3600))
    write(out/'time_to_target.csv',costs)
    print('Saved summaries to outputs/summary')

if __name__=='__main__':main()
