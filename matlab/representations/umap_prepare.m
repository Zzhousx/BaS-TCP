function prep=umap_prepare(Features,params)
d=4;K=getf(params,'umapK',3);
X=double(Features);mu=mean(X,1);sd=std(X,0,1);sd(sd==0)=1;X=(X-mu)./sd;
n=size(X,1);K=max(2,min(K,n-1));sq=sum(X.^2,2);
D2=max(sq+sq'-2*(X*X'),0);D2(1:n+1:end)=inf;Dm=sqrt(D2);
[sortedD,ord]=sort(Dm,2,'ascend');nbr=ord(:,1:K);dnbr=sortedD(:,1:K);
clear D2 Dm sortedD ord X
rho=dnbr(:,1);target=log2(K);sigma=ones(n,1);
for i=1:n
 lo=1e-6;hi=1e6;
 for it=1:64
  mid=(lo+hi)/2;s=sum(exp(-max(dnbr(i,:)-rho(i),0)/mid));
  if s>target,hi=mid;else,lo=mid;end
 end
 sigma(i)=(lo+hi)/2;
end
P=zeros(n,n);
for i=1:n,P(i,nbr(i,:))=exp(-max(dnbr(i,:)-rho(i),0)/sigma(i));end
W=P+P'-P.*P';W=(W+W')/2;clear P
deg=sum(W,2);Dinv=1./sqrt(max(deg,1e-12));
L=eye(n)-(Dinv.*W).*Dinv';L=(L+L')/2;
[V,Dg]=eig(L);[~,idx]=sort(real(diag(Dg)),'ascend');Z=real(V(:,idx(2:d+1)));
if max(abs(Z(:)))>0,Z=10*Z/max(abs(Z(:)));end
[ei,ej,ew]=find(triu(W,1));
prep=struct('Z0',Z,'ei',ei,'ej',ej,'ew',ew,'n',n,'k',K,'dim',d);
end
function v=getf(s,f,dv)
if isfield(s,f)&&~isempty(s.(f)),v=s.(f);else,v=dv;end
end
