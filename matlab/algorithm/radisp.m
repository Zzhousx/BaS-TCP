function [W,obj,indexW,Ypre] = radisp(X,k,alpha,l)
param=struct('NITER',100,'c',2);k2=alpha;k1=l;
NITER = param.NITER;
alpha = k2;
MAX_WEIGHT = 1e5;

X = X';
[d, n] = size(X);
c = param.c;
num = k1;
X = X - mean(X, 2);

options = [];
options.ReducedDim = num;
[W, ~] = PCA(X', options);
q = ones(n, 1);
q1 = ones(n, 1);

obj = [];
for iter = 1:NITER
    M = ((W'*X) .* sqrt(q'))';

    Ypre = density_kmeans(M, c);
    Y = full(ind2vec(Ypre', c))';

    if size(Y, 2) < c
        fprintf('Warning: custom_kmeans resulted in less than %d clusters. Re-running with standard kmeans.\n', c);
        Ypre = kmeans(M, c, 'replicates', 10, 'EmptyAction', 'singleton');
        Y = full(ind2vec(Ypre', c))';
        if size(Y, 2) < c, break; end
    end

    T1 = (W'*X) * Y;
    T2 = pinv(Y'*Y);
    T3 = (T1 * T2) * Y';
    E1 = ((W'*X) - T3)';

    qj = sqrt(sum(E1.*E1,2) + eps);
    q_new_diag = 0.5./qj;
    q_new_diag(q_new_diag > MAX_WEIGHT) = MAX_WEIGHT;

    q = (1-alpha)*q + alpha*q_new_diag;

    E2 = (W'*X)';
    q1j = sqrt(sum(E2.*E2,2) + eps);
    q1_new_diag = 0.5./q1j;
    q1_new_diag(q1_new_diag > MAX_WEIGHT) = MAX_WEIGHT;

    q1 = (1-alpha)*q1 + alpha*q1_new_diag;

    A = (X .* q') * Y;
    B = pinv(Y'*Y);
    C = (X * Y)';
    S_b = A * B * C + eps*eye(d);
    S_t = (X .* q1') * X' + eps*eye(d);

    W = Solve_W_Subproblem(S_b, S_t, W, num, k);

    final_Sp_tr = trace(W'*S_t*W);
    final_Sq_tr = trace(W'*S_b*W);
    if abs(final_Sq_tr) < 1e-9 , break; end
    obj(iter) = final_Sp_tr / final_Sq_tr;

    if iter > 1 && abs(obj(iter) - obj(iter-1)) < 1e-6
        obj = obj(1:iter);
        break;
    end
end
[~, indexW] = sort(sum(W.*W, 2), 'descend');
indexW = indexW(1:k);
end
