function [Z, info] = repr_PCA(Features, params)

    if nargin < 2 || isempty(params), params = struct(); end
    d = getf(params,'pcaDim', 5);

    X = double(Features);
    [m, n] = size(X);

    for i = 1:n
        lo = min(X(:,i)); hi = max(X(:,i));
        rg = hi - lo;
        if rg == 0, rg = 1; end
        X(:,i) = (X(:,i) - lo) / rg;
    end
    [~, scores, latent] = pca(X);
    d = max(1, min(d, size(scores,2)));
    Z = scores(:, 1:d);

    tot = sum(latent);
    if tot <= 0, tot = 1; end
    info = struct('dim', d, 'explainedVarianceRatio', latent(1:d)/tot);
end

function v = getf(s, f, dv)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dv; end
end
