function [order,seconds] = greedy_tcp(X,costs,metric)
if nargin<3, metric='seuclidean'; end
t=tic; n=size(X,1); D=squareform(pdist(X,metric));
mx=max(D(:)); [r,~]=find(D==mx); a=r(1); b=r(2);
order=zeros(n,1); order(1:2)=[a;b]; chosen=false(n,1); chosen([a,b])=true;
sums=D(:,a)+D(:,b);
for pos=3:n
    score=sums./costs(:); score(chosen)=-Inf;
    [~,best]=max(score); order(pos)=best; chosen(best)=true;
    sums=sums+D(:,best);
end
seconds=toc(t);
end
