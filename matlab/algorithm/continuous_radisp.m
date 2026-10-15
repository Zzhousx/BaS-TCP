function R=continuous_radisp(X,hp,rho)
t0=tic;X=X';[d,n]=size(X);X=X-mean(X,2);options=struct('ReducedDim',hp.l);[W,~]=PCA(X',options);
q=ones(n,1);q1=q;history=[];info={};lambda=NaN;
for iter=1:100
 M=((W'*X).*sqrt(q'))';Ypre=density_kmeans(M,2);
 if numel(unique(Ypre))<2 || any(~isfinite(Ypre)),rng(0,'twister');Ypre=kmeans(M,2,'Replicates',10,'EmptyAction','singleton');end
 Y=full(sparse(1:n,Ypre(:)',1,n,2));B=pinv(Y'*Y);
 E=((W'*X)-(W'*X)*Y*B*Y')';qnew=min(1e5,.5./sqrt(sum(E.^2,2)+eps));q=(1-hp.alpha)*q+hp.alpha*qnew;
 E=(W'*X)';qnew=min(1e5,.5./sqrt(sum(E.^2,2)+eps));q1=(1-hp.alpha)*q1+hp.alpha*qnew;
 Sb=((X.*q')*Y)*B*(X*Y)'+eps*eye(d);St=(X.*q1')*X'+eps*eye(d);
 if iter==1
  initialRatio=trace(W'*St*W)/trace(W'*Sb*W);lambda=rho*initialRatio/sum(sqrt(sum(W.^2,2)));
  assert(isfinite(lambda)&&lambda>=0);
 end
 [W,I]=group_manpg(St,Sb,W,lambda,300,1e-5);info{end+1}=I;
 assert(I.orthogonality_error<1e-7&&all(isfinite(W(:))),'Invalid manifold solution');
 assert(~strcmp(I.status,'tangent_prox_failed'),'Tangent proximal solve failed');
 obj=trace(W'*St*W)/trace(W'*Sb*W)+lambda*sum(sqrt(sum(W.^2,2)));history(end+1)=obj;
 if iter>1&&abs(history(end)-history(end-1))<1e-6,break;end
 fprintf('GROUP outer rho%g iter%d inner%d %s support%d\n',rho,iter,I.iterations,I.status,I.active_rows);
end
[~,ix]=sort(sum(W.^2,2),'descend');ix=ix(1:hp.k);V=W(ix,:);
if rank(V)<hp.l,[V,~]=eig((Sb(ix,ix)+Sb(ix,ix)')/2);V=V(:,end-hp.l+1:end);else,[V,~]=qr(V,0);end
V=Solve_W_Subproblem(Sb(ix,ix),St(ix,ix),V,hp.l,hp.k);
Wmatched=zeros(d,hp.l);Wmatched(ix,:)=V;
assert(sum(sqrt(sum(Wmatched.^2,2))>1e-8)<=hp.k&&norm(Wmatched'*Wmatched-eye(hp.l),'fro')<1e-7);
R=struct('W',W,'Wmatched',Wmatched,'matched_indices',ix,'Ypre',Ypre,'lambda',lambda, ...
 'outer_objective_history',history,'inner_info',{info},'outer_iterations',iter,'elapsed',toc(t0));
end
