function run_components(datasets,methods,seeds)
if nargin<1,datasets=1:3;end
if nargin<2,methods={'raw','pca','umap','vae','lle','isa','greedy','ga'};end
if nargin<3,seeds=1:30;end
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'outputs','components');if ~isfolder(out),mkdir(out);end
for id=datasets
 D=load_dataset(id);hp=default_config(D.X);prep=[];
 rp=struct('k1',hp.l,'lleK',12,'lleDim',hp.l,'lleReg',1000,'isaDim',hp.l,'isaReg',1e-6, ...
  'vaeH',32,'vaeEpochs',600,'vaeLr',.01,'umapK',10,'umapEpochs',50,'umapMinDist',.1, ...
  'umapLr',.8,'umapNegSamples',3,'umapDim',hp.l);
 H=load(fullfile(root,'results','effectiveness',sprintf('D%d_full_bas_s00.mat',id)),'R');H=H.R;
 for mi=1:numel(methods)
  method=methods{mi};ss=0;if ismember(method,{'umap','vae','ga'}),ss=seeds;end
  if ismember(method,{'greedy','ga'}),maxNumCompThreads(2);else,maxNumCompThreads(4);end
  for seed=ss
   file=fullfile(out,sprintf('D%d_%s_s%02d.mat',id,method,seed));if isfile(file),continue;end
   if strcmp(method,'umap')&&isempty(prep),prep=umap_prepare(D.X,rp);end
   rng(seed,'twister');t=tic;
   switch method
    case 'raw',Z=D.X;s=std(Z,0,1);s(s==0)=1;Z=(Z-mean(Z,1))./s;
    case 'pca',Z=repr_PCA(D.X,rp);
    case 'lle',Z=repr_LLE(D.X,rp);
    case 'isa',Z=repr_ISA(D.X,D.y,rp);
    case 'vae',Z=repr_VAE(D.X,rp);
    case 'umap',Z=umap_embed(prep,rp);
    case {'greedy','ga'},Z=H.Z;
    otherwise,error('Unknown representation/search method');
   end
   embedding=toc(t);t=tic;
   if strcmp(method,'greedy'),order=greedy_tcp(Z,D.costs);
   elseif strcmp(method,'ga'),order=ga_tcp(Z,D.costs,true);
   else,order=balanced_prioritize(Z,D.ranking_costs,hp.eta,hp.beta);end
   R=struct('dataset',id,'method',method,'seed',seed,'order',order(:),'score',score_order(order,D.y,D.costs), ...
    'Z',Z,'embedding_seconds',embedding,'search_seconds',toc(t),'parameters',rp,'threads',maxNumCompThreads);
   save(file,'R','-v7');fprintf('D%d %s seed%d %.12f\n',id,method,seed,R.score);
  end
 end
end
end
