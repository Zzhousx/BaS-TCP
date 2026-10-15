function W_opt = Solve_W_Subproblem(Sb, St, W_init, l, k)

max_ratio_iter = 20; % Max iterations for the outer ratio loop
W = W_init;
[d, ~] = size(Sb);

for i = 1:max_ratio_iter
    tr_b = trace(W' * Sb * W);
    tr_t = trace(W' * St * W);
    if abs(tr_t) < 1e-9, break; end
    lambda = tr_b / tr_t;

    A = Sb - lambda * St;

    A = (A + A') / 2;
    min_eig = min(eig(A));
    if min_eig < 0
        A = A - min_eig * eye(d);
    end

    W_new = Solve_Sparse_Subspace(A, W, l, k);

    if norm(W_new - W, 'fro') / norm(W, 'fro') < 1e-5
        W = W_new;
        break;
    end
    W = W_new;
end
W_opt = W;
end
