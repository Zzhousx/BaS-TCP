function hp=default_config(X)
d=size(X,2);r=rank(X-mean(X,1));k=ceil(d/2);
hp=struct('k',k,'l',min([10,k,r]),'alpha',0.75,'eta',1,'beta',1);
end
