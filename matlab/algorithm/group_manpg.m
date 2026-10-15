function [W,info]=group_manpg(St,Sb,W,lambda,maxiter,tol)
if nargin<5,maxiter=300;end
if nargin<6,tol=1e-5;end
St=(St+St')/2;Sb=(Sb+Sb')/2;l=size(W,2);d=size(W,1);
bases={};
for a=1:l
 E=zeros(l);E(a,a)=1;bases{end+1}=E;
 for b=a+1:l,E=zeros(l);E(a,b)=1/sqrt(2);E(b,a)=1/sqrt(2);bases{end+1}=E;end
end
[value,ratio,G,den]=objective(W,St,Sb,lambda);
t=1/max(1e-8,2*(norm(St,2)+abs(ratio)*norm(Sb,2))/den);t0=t;
hist=value;S=zeros(l);residuals=[];steps=[];status='iteration_limit';station=Inf;
for it=1:maxiter
 success=false;
 for retry=1:10
  [V,S,dualres,dualiter]=tangent_prox(W,G,t,lambda,S,bases);
  if dualres<1e-7 && all(isfinite(V(:))),success=true;break;end
  t=t/2;S=zeros(l);
 end
 if ~success,status='tangent_prox_failed';break;end
 D=V-W;station=norm(D,'fro')/t;residuals(end+1)=dualres;
 if station<=tol*(1+norm(G,'fro')),status='stationary_tolerance';break;end
 alpha=1;accepted=false;
 for ls=1:35
  candidate=polar_rows(W+alpha*D);
  [nv,nr,nG,nd]=objective(candidate,St,Sb,lambda);
  if isfinite(nv) && nv<=value-1e-4*alpha*norm(D,'fro')^2/t
   accepted=true;break;
  end
  alpha=alpha/2;
 end
 if ~accepted,status='line_search_limit';break;end
 W=candidate;value=nv;ratio=nr;G=nG;den=nd;
 hist(end+1)=value;steps(end+1)=alpha;
 if alpha==1,t=min(t*1.1,20*t0);end
end
info=struct('status',status,'iterations',it,'objective_history',hist,'stationarity_residual',station, ...
 'max_tangent_residual',max([0,residuals]),'orthogonality_error',norm(W'*W-eye(l),'fro'), ...
 'exact_zero_rows',sum(all(W==0,2)),'active_rows',sum(sqrt(sum(W.^2,2))>1e-8), ...
 'steps',steps,'last_dual_iterations',dualiter,'lambda',lambda,'ratio',ratio);
end
function [value,r,G,den]=objective(W,A,B,lambda)
den=trace(W'*B*W);
if den<=1e-12,value=Inf;r=NaN;G=NaN(size(W));return;end
r=trace(W'*A*W)/den;
value=r+lambda*sum(sqrt(sum(W.^2,2)));
G=2*(A*W-r*B*W)/den;
end
function [V,S,res,it]=tangent_prox(W,G,t,lambda,S,bases)
m=numel(bases);l=size(W,2);tau=t*lambda;reg=1e-10;
for it=1:80
 U=W-t*G-W*S;
 [V,active,fac,r]=group_prox(U,tau);
 R=(W'*V+V'*W)/2-eye(l);v=pack(R,bases);res=norm(v);
 if res<1e-10,return;end
 J=zeros(m);
 for j=1:m
  DU=-W*bases{j};DV=DU.*fac;
  if tau>0
   ix=find(active);DV(ix,:)=DV(ix,:)+U(ix,:).*((tau./r(ix).^3).*sum(U(ix,:).*DU(ix,:),2));
  end
  JR=(W'*DV+DV'*W)/2;J(:,j)=pack(JR,bases);
 end
 J=(J+J')/2;delta=(-J+reg*eye(m))\v;
 DS=zeros(l);for j=1:m,DS=DS+delta(j)*bases{j};end
 accepted=false;a=1;
 for ls=1:25
  NS=S+a*DS;NV=group_prox(W-t*G-W*NS,tau);
  NR=(W'*NV+NV'*W)/2-eye(l);nres=norm(pack(NR,bases));
  if nres<res*(1-1e-4*a),accepted=true;break;end
  a=a/2;
 end
 if accepted,S=NS;reg=max(1e-12,reg/2);else,reg=max(1e-8,reg*100);end
end
U=W-t*G-W*S;V=group_prox(U,tau);R=(W'*V+V'*W)/2-eye(l);res=norm(pack(R,bases));
end
function [V,active,fac,r]=group_prox(U,tau)
r=sqrt(sum(U.^2,2));active=r>tau;fac=zeros(size(r));
if tau==0,fac(:)=1;else,fac(active)=1-tau./r(active);end
V=U.*fac;
end
function v=pack(M,bases)
v=zeros(numel(bases),1);for j=1:numel(bases),v(j)=sum(sum(M.*bases{j}));end
end
function W=polar_rows(V)
[Q,E]=eig((V'*V+(V'*V)')/2);e=diag(E);
if min(e)<=1e-12,W=NaN(size(V));else,W=V*(Q*diag(1./sqrt(e))*Q');end
end
