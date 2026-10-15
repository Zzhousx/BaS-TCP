import argparse,csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from summarize import ROOT,NAMES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['results','outputs'],default='results');args=ap.parse_args()
    directory=ROOT/args.source/'effectiveness';groups={}
    for p in sorted(directory.glob('*.mat')):
        r=loadmat(p,simplify_cells=True)['R']
        if str(r.get('view','full'))!='full':continue
        groups.setdefault((int(r['dataset']),str(r['method'])),[]).append(r)
    out=ROOT/'outputs/summary';out.mkdir(parents=True,exist_ok=True)
    with (out/'curves.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['Dataset','Method','Budget','MeanFailingFraction']);w.writeheader();grid=np.linspace(0,1,2001)
        for (d,method),records in sorted(groups.items()):
            D=loadmat(ROOT/f'data/Dataset{d}.mat',squeeze_me=True);y=D['y'].astype(bool);c=D['costs'];curves=[]
            for r in records:
                order=np.asarray(r['order']).ravel().astype(int)-1;x=np.cumsum(c[order])/c.sum();v=np.cumsum(y[order])/y.sum()
                if d==3:x=np.r_[0,x];v=np.r_[0,v]
                curves.append(np.interp(grid,x,v,left=np.nan))
            a=np.asarray(curves);valid=np.all(np.isfinite(a),axis=0);mean=a[:,valid].mean(axis=0)
            w.writerows(dict(Dataset=d,Method=NAMES[method],Budget=x,MeanFailingFraction=z) for x,z in zip(grid[valid],mean))

if __name__=='__main__':main()
