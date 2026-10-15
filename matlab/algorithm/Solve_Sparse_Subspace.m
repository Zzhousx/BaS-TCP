function W_opt = Solve_Sparse_Subspace(A, W_init, l, k)

max_mm_iter = 20; % Max iterations for the MM loop
W = W_init;
[d, ~] = size(A);

for t = 1:max_mm_iter
    WTAW = W' * A * W;
    At = A * W * pinv(WTAW) * W' * A;


    diag_At = diag(At);
    [~, I_t] = sort(diag_At, 'descend');
    I_t = I_t(1:k); % These are the indices of the selected features

    Ht = zeros(d, k);
    for j = 1:k
        Ht(I_t(j), j) = 1;
    end

    A_sub = Ht' * At * Ht;
    A_sub = (A_sub + A_sub')/2;

    if k < l
         Zt = zeros(k, l);
    else
        [Zt, ~] = eigs(A_sub, l); % Zt is k x l
    end


    W_new = Ht * Zt; % W_new is d x l

    if norm(W_new - W, 'fro') / norm(W, 'fro') < 1e-5
        W = W_new;
        break;
    end
    W = W_new;
end
W_opt = W;
end
