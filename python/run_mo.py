import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse, random, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

class PrefixCrossover(Crossover):
    def __init__(self, probability=0.8):
        super().__init__(2, 1, prob=probability)
    def _do(self, problem, X, **kwargs):
        _, count, n = X.shape
        Y = np.empty((1, count, n), dtype=int)
        for i in range(count):
            a, b = X[:, i, :]
            cut = random.randint(0, n - 1)
            head = a[:cut]
            mask = np.zeros(n, dtype=bool)
            mask[head] = True
            Y[0, i] = np.concatenate((head, b[~mask[b]]))
        return Y

class HybridMutation(Mutation):
    def __init__(self):
        super().__init__(prob=1.0)
    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        n = Y.shape[1]
        for row in Y:
            a, b = random.sample(range(n), 2)
            p = random.random()
            if p <= .33:
                row[a], row[b] = row[b], row[a]
            elif p <= .66:
                lo, hi = sorted((a, b))
                row[lo:hi+1] = row[lo:hi+1][::-1]
            else:
                row[:] = np.insert(np.delete(row, a), b, row[a])
        return Y

class TCPProblem(Problem):
    def __init__(self, distance, costs):
        super().__init__(n_var=len(costs), n_obj=2, vtype=int)
        self.distance, self.costs = distance, costs
        self.weights = 1.0 / np.arange(1, len(costs))
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = (self.costs[X[:, :-1]] * self.weights).sum(axis=1)
        f2 = -(self.distance[X[:, :-1], X[:, 1:]] * self.weights).sum(axis=1)
        out["F"] = np.column_stack((f1, f2))

class Progress(Callback):
    def notify(self, algorithm):
        if algorithm.n_gen % 400 == 0:
            print(f"MO progress generation {algorithm.n_gen}", flush=True)

def metric(order, y, costs):
    total, faults = costs.sum(), y.sum()
    return 1 - np.cumsum(costs[order])[y[order]].sum() / (total*faults) + costs[order][y[order]].sum() / (2*total*faults)

def score_order(order, y, costs):
    if np.all(costs == 1):
        return metric(order, y, costs)
    return np.trapezoid(np.cumsum(y[order]), np.cumsum(costs[order])) / (costs.sum()*y.sum())


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',type=int,nargs='+',default=[1,2,3],choices=[1,2,3])
    parser.add_argument('--view',default='full',choices=['full','feature10'])
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(1,31)))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'outputs/effectiveness';out.mkdir(parents=True,exist_ok=True)
    for id in args.datasets:
        D=loadmat(root/f'data/Dataset{id}.mat',squeeze_me=True)
        X=D['X'];y=D['y'].astype(bool);costs=D['ranking_costs'];eval_costs=D['costs']
        if args.view=='feature10':
            assert id==1
            X=X[:,14:24]
        t=time.perf_counter();span=np.ptp(X,axis=0);span[span==0]=1
        Z=(X-X.min(axis=0))/span;Z-=Z.mean(axis=0)
        _,_,vt=np.linalg.svd(Z,full_matrices=False);Z=Z@vt[:10].T
        distance=squareform(pdist(Z,'euclidean'));preprocess=time.perf_counter()-t
        problem=TCPProblem(distance,costs)
        for seed in args.seeds:
            assert 1<=seed<=30
            file=out/f'D{id}_{args.view}_mo_s{seed:02d}.mat'
            if file.exists():continue
            random.seed(seed);np.random.seed(seed)
            algorithm=NSGA2(pop_size=100,sampling=PermutationRandomSampling(),crossover=PrefixCrossover(.8),mutation=HybridMutation(),eliminate_duplicates=True)
            t=time.perf_counter()
            result=minimize(problem,algorithm,('n_gen',4000),seed=seed,verbose=False,callback=Progress(),save_history=False)
            low=result.F.min(axis=0);sp=np.ptp(result.F,axis=0);sp[sp==0]=1
            pick=int(np.argmin(np.linalg.norm((result.F-low)/sp,axis=1)));order=result.X[pick].astype(int)
            assert np.array_equal(np.sort(order),np.arange(len(y))) and result.algorithm.n_gen-1==4000
            R=dict(dataset=id,view=args.view,method='mo',seed=seed,order=(order+1).reshape(-1,1),score=float(score_order(order,y,eval_costs)),
                seconds=time.perf_counter()-t,preprocess_seconds=preprocess,threads=4,population=100,generations=4000,crossover=.8)
            savemat(file,{'R':R},do_compression=True);print(f'D{id} MO-TCP seed{seed} {R["score"]:.12f}',flush=True)

if __name__=='__main__':run()
