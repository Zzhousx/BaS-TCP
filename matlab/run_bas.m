function R=run_bas(X,costs)
rng(0,'twister');hp=default_config(X);t=tic;
[W,history,indices,clusters]=radisp(X,hp.k,hp.alpha,hp.l);learning=toc(t);
Z=X*W;t=tic;order=balanced_prioritize(Z,costs,hp.eta,hp.beta);search=toc(t);
R=struct('W',W,'Z',Z,'history',history,'indices',indices,'clusters',clusters, ...
 'order',order(:),'learning_seconds',learning,'search_seconds',search,'seconds',learning+search,'parameters',hp);
end
