function [Z, info] = repr_LLE(Features, params)


    if nargin < 2 || isempty(params), params = struct(); end
    K   = getf(params,'lleK', 5);
    d   = 10;
    reg = getf(params,'lleReg', 1e+3);
    doStd = getf(params,'lleStandardize', false);

    X = double(Features);
    n = size(X,1);


    sq = sum(X.^2, 2);
    D2 = max(sq + sq' + 2*(X*X'), 0);
    D2(1:n+1:end) = inf;
    [~, ord] = sort(D2, 2, 'ascend');
    nbr = ord(:, 1:K);

    W = zeros(n, n);
    for i = 1:n
        Zi = X(nbr(i,:), :) - X(i, :);
        C  = Zi * Zi';
        tr = trace(C);
        if tr > 0
            C = C + reg * tr * eye(K);
        else
            C = C + reg * eye(K);
        end
        w = C \ ones(K,1);
        w = w / sum(w);
        W(i, nbr(i,:)) = w';
    end


    M = eye(n) - W;
    M = M' * M;
    M = (M + M') / 2;
    [V, Dg] = eig(M);
    [ev, idx] = sort(diag(Dg), 'ascend');
    V = V(:, idx);
    Z = V(:, 2:d+1);
    Z = Z * sqrt(n);

    info = struct('k', K, 'dim', d, 'reg', reg, ...
                  'eigenvalues', ev(1:min(d+2, numel(ev))));
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
