function run_effectiveness(datasets,methods,seeds,view)
if nargin<1,datasets=1:3;end
if nargin<2,methods={'bas','greedy','original','strap','random','art','cluster','spectre','rl','least','ga'};end
if nargin<3,seeds=1:30;end
if nargin<4,view='full';end
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'outputs','effectiveness');if ~isfolder(out),mkdir(out);end
for id=datasets
 D=load_dataset(id,view);p=baseline_config(id,size(D.X,2));
 for mi=1:numel(methods)
  method=methods{mi};ss=seeds;if ismember(method,{'bas','greedy','original','strap'}),ss=0;end
  for seed=ss
   file=fullfile(out,sprintf('D%d_%s_%s_s%02d.mat',id,view,method,seed));if isfile(file),continue;end
   if strcmp(method,'bas'),maxNumCompThreads(2);else,maxNumCompThreads(4);end
   rng(seed,'twister');t=tic;R=struct();
   switch method
    case 'bas',R=run_bas(D.X,D.ranking_costs);order=R.order;
    case 'greedy',order=greedy_tcp(D.X,D.costs);
    case 'original',order=(1:numel(D.y))';
    case 'strap',order=strap(D.X,p,struct(),D.ranking_costs);
    case 'random',order=randperm(numel(D.y))';
    case 'art',order=art_tcp(D.X,p.artCandPool,D.ranking_costs,p.artCostAware);
    case 'cluster',order=cluster_tcp(D.X,D.ranking_costs,p);
    case 'spectre',order=spectre_adapted(D.X,D.ranking_costs,p,struct());
    case 'rl',order=rl_static(D.X,D.ranking_costs,D.y,p,[]);
    case 'least',order=least_static(D.X,D.y,p,[]);
    case 'ga',[order,~,R.ga_output]=ga_tcp(D.X,D.costs,false,view);
    otherwise,error('Unknown method');
   end
   R.seconds=toc(t);R.dataset=id;R.view=view;R.method=method;R.seed=seed;R.order=order(:);
   R.score=score_order(order,D.y,D.costs);R.threads=maxNumCompThreads;
   if ~strcmp(method,'bas'),R.parameters=p;end
   save(file,'R','-v7');fprintf('D%d %s %s seed%d %.12f\n',id,view,method,seed,R.score);
  end
 end
end
end
