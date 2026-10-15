function [order,elapsed,output]=ga_tcp(X,costs,isLandscape,view)
if nargin<3,isLandscape=false;end
if nargin<4,view='full';end
t=tic;
if isLandscape,Z=X;else
 [~,Z]=pca(X);Z=Z(:,1:min(10,size(Z,2)));
 if strcmp(view,'feature10'),s=std(Z,0,1);Z=Z(:,s>1e-12*max(s));end
end
A=squareform(pdist(Z,'seuclidean'));assert(all(isfinite(A(:))));
c=costs(:);n=numel(c);
opt=optimoptions('ga','PopulationType','custom','PopulationSize',100,'MaxGenerations',4000, ...
 'MaxStallGenerations',4001,'FunctionTolerance',0,'CrossoverFraction',0.8, ...
 'CreationFcn',@(len,ff,options)create_pop(len,options,c),'SelectionFcn',@selectionroulette, ...
 'CrossoverFcn',@permutationCrossover,'MutationFcn',@permutationMutation,'Display','off','PlotFcn',[]);
[order,~,~,output]=ga(@(perm)fitness_vector(perm,A,c),n,[],[],[],[],[],[],[],[],opt);
elapsed=toc(t);order=order(:);assert(output.generations==4000);
end
function f=fitness_vector(x,A,c)
x=x(:)';n=numel(x);previous=[x(2),x(1:end-1)];d=A(sub2ind(size(A),previous,x));f=-sum(d./c(x)'./(1:n));
end
function pop=create_pop(n,opt,c)
pop=zeros(sum(opt.PopulationSize),n);for r=1:size(pop,1)-1,pop(r,:)=randperm(n);end
[~,ix]=sort(c,'ascend');pop(end,:)=ix(:)';
end
