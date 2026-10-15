function [Z,info]=umap_embed(prep,params)
epochs=getf(params,'umapEpochs',50);lr0=getf(params,'umapLr',.8);
nNeg=getf(params,'umapNegSamples',3);minD=getf(params,'umapMinDist',.1);
Z=prep.Z0;ei=prep.ei;ej=prep.ej;ew=prep.ew;n=prep.n;
if isempty(ei),info=struct('k',prep.k,'dim',prep.dim,'edges',0);return;end
a=1;b=1;if minD>0,a=1/(1+minD);end
nE=numel(ei);
for ep=1:epochs
 alpha=lr0*(1-(ep-1)/epochs);perm=randperm(nE);
 for t=perm
  if rand>ew(t),continue;end
  i=ei(t);j=ej(t);diff=Z(i,:)-Z(j,:);dist2=sum(diff.^2)+1e-12;
  gcoef=(-2*a*b*dist2^(b-1))/(1+a*dist2^b);g=max(min(gcoef*diff,4),-4);
  Z(i,:)=Z(i,:)+alpha*g;Z(j,:)=Z(j,:)-alpha*g;
  for s=1:nNeg
   kNeg=randi(n);if kNeg==i,continue;end
   diff=Z(i,:)-Z(kNeg,:);dist2=sum(diff.^2)+1e-12;
   gcoef=(2*b)/((1e-3+dist2)*(1+a*dist2^b));g=max(min(gcoef*diff,4),-4);
   Z(i,:)=Z(i,:)+alpha*g;
  end
 end
end
info=struct('k',prep.k,'dim',prep.dim,'edges',nE);
end
function v=getf(s,f,dv)
if isfield(s,f)&&~isempty(s.(f)),v=s.(f);else,v=dv;end
end
