function run_timing(datasets)
if nargin<1,datasets=1:3;end
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'outputs','timing');if ~isfolder(out),mkdir(out);end
maxNumCompThreads(4);rows={};
for id=datasets
 D=load_dataset(id);run_bas(D.X,D.ranking_costs);greedy_tcp(D.X,D.costs);
 for rep=1:15
  if mod(rep,2),R=run_bas(D.X,D.ranking_costs);[~,tg]=greedy_tcp(D.X,D.costs);
  else,[~,tg]=greedy_tcp(D.X,D.costs);R=run_bas(D.X,D.ranking_costs);end
  rows(end+1,:)={id,rep,R.learning_seconds,R.search_seconds,R.seconds,tg};
  writetable(cell2table(rows,'VariableNames',{'Dataset','Repetition','Learning','Search','Total','Greedy'}),fullfile(out,'timing.csv'));
 end
end
end
