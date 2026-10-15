function run_sparsity(datasets)
if nargin<1,datasets=1:3;end
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'outputs','sparsity');if ~isfolder(out),mkdir(out);end
maxNumCompThreads(2);
for id=datasets
 D=load_dataset(id);hp=default_config(D.X);cases=struct([]);
 for rho=[0 .01 .1 1 10],cases=[cases;evaluate_case(D.X,hp,rho)];end
 for refinement=1:8
  if any([cases.valid]&[cases.support]==hp.k),break;end
  [~,ix]=sort([cases.rho]);cases=cases(ix);bounds=[];
  for j=1:numel(cases)-1
   a=cases(j);b=cases(j+1);
   crossing=(a.support>hp.k&&b.support<=hp.k)||(b.support>hp.k&&a.support<=hp.k);
   if a.rho>0&&crossing&&b.rho/a.rho>1.01,bounds(end+1,:)=[j,log(b.rho/a.rho),a.rho];end
  end
  if isempty(bounds),break;end
  [~,jj]=sortrows([-bounds(:,2),bounds(:,3)],[1 2]);j=bounds(jj(1),1);
  cases=[cases;evaluate_case(D.X,hp,sqrt(cases(j).rho*cases(j+1).rho))];
 end
 ix=find([cases.valid]&[cases.support]<=hp.k);assert(~isempty(ix));
 [~,s]=sortrows([-[cases(ix).support]',[cases(ix).rho]'],[1 2]);choice=cases(ix(s(1)));R=choice.result;
 save(fullfile(out,sprintf('D%d_feature_choice.mat',id)),'choice','cases','-v7');
 R.order=balanced_prioritize(D.X*R.W,D.ranking_costs,hp.eta,hp.beta);
 R.matched_order=balanced_prioritize(D.X*R.Wmatched,D.ranking_costs,hp.eta,hp.beta);
 R.score=score_order(R.order,D.y,D.costs);R.matched_score=score_order(R.matched_order,D.y,D.costs);
 R.dataset=id;R.rho=choice.rho;R.support=choice.support;R.parameters=hp;
 save(fullfile(out,sprintf('D%d.mat',id)),'R','-v7');
end
end
function C=evaluate_case(X,hp,rho)
rng(0,'twister');R=continuous_radisp(X,hp,rho);I=R.inner_info{end};
C=struct('rho',rho,'support',sum(sqrt(sum(R.W.^2,2))>1e-8), ...
 'valid',strcmp(I.status,'stationary_tolerance')&&I.orthogonality_error<1e-7&&I.max_tangent_residual<1e-7,'result',R);
end
