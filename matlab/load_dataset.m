function D=load_dataset(id,view)
if nargin<2,view='full';end
root=fileparts(fileparts(mfilename('fullpath')));
D=load(fullfile(root,'data',sprintf('Dataset%d.mat',id)));
D.id=id;D.view=view;D.y=logical(D.y(:));D.costs=D.costs(:);D.ranking_costs=D.ranking_costs(:);
if strcmp(view,'feature10')
 assert(id==1);D.X=D.X(:,15:24);
elseif ~strcmp(view,'full'),error('Unknown feature view');end
assert(size(D.X,1)==numel(D.y)&&all(isfinite(D.X(:)))&&all(D.costs>0));
end
