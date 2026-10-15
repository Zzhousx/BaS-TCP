function order = strap(Features, params, attribs, Cost)

    if nargin < 2 || isempty(params), params = struct(); end
    if nargin < 3, attribs = struct(); end
    if nargin < 4 || isempty(Cost), Cost = ones(size(Features,1),1); end
    Cost = Cost(:);
    costAware = getf(params,'dengCostAware', false);
    nBins   = getf(params,'dengBins', 5);
    mode    = getf(params,'dengMode','incremental');
    binMeth = getf(params,'dengBinMethod','quantile');
    dynW    = getf(params,'dengDynamicWeights', false);
    catCols = [];
    if isfield(attribs,'catCols'), catCols = attribs.catCols(:)'; end

    X = double(Features);
    [N, d] = size(X);

    V = [];
    for j = 1:d
        col = X(:, j);
        u = unique(col);
        if ismember(j, catCols) || numel(u) <= nBins
            vals = u(u ~= 0);
            for t = 1:numel(vals)
                V(:, end+1) = double(col == vals(t)); %#ok<AGROW>
            end
        else
            if strcmpi(binMeth,'quantile')
                edges = unique(quantile(col, linspace(0,1,nBins+1)));
            else
                edges = linspace(min(col), max(col), nBins+1);
            end
            if numel(edges) < 2, continue; end
            edges(1) = -inf; edges(end) = inf;
            for t = 1:numel(edges)-1
                V(:, end+1) = double(col > edges(t) & col <= edges(t+1)); %#ok<AGROW>
            end
        end
    end
    Q = size(V, 2);
    if Q == 0, order = (1:N)'; return; end

    occ = sum(V, 1);
    w = zeros(1, Q);
    nz = occ > 0;
    w(nz) = N ./ occ(nz);
    sw = sum(w);
    if sw > 0, w = w / sw; end

    [~, ~, grpId] = unique(V, 'rows', 'stable');
    isRep = false(N,1);
    for g = 1:max(grpId)
        mem = find(grpId == g);
        isRep(mem(1)) = true;
    end

    if strcmpi(mode, 'total')
        rarity = V * w';
        rarity(~isRep) = rarity(~isRep) * 0.5;
        if costAware, rarity = rarity ./ max(Cost, 1e-9); end
        [~, order] = sort(rarity, 'descend');
        order = order(:); return;
    end

    order     = zeros(N, 1);
    remaining = true(N, 1);
    covered   = false(1, Q);
    Vl = logical(V);

    for step = 1:N
        if dynW && any(remaining)
            occR = sum(V(remaining,:), 1);
            wR = zeros(1, Q); nzR = occR > 0;
            wR(nzR) = sum(remaining) ./ occR(nzR);
            swR = sum(wR); if swR > 0, wR = wR / swR; end
        else
            wR = w;
        end

        newGain = zeros(N, 1);
        uncov = ~covered;
        if any(uncov)
            newGain(remaining) = Vl(remaining, uncov) * wR(uncov)';
        end
        if max(newGain) <= 0
            covered(:) = false; uncov = ~covered;
            newGain(:) = 0;
            newGain(remaining) = Vl(remaining, uncov) * wR(uncov)';
        end
        dupMask = remaining & ~isRep;
        newGain(dupMask) = newGain(dupMask) * 0.5;
        if costAware
            newGain(remaining) = newGain(remaining) ./ max(Cost(remaining), 1e-9);
        end
        newGain(~remaining) = -inf;
        [~, best] = max(newGain);
        order(step) = best;
        remaining(best) = false;
        covered = covered | Vl(best, :);
    end
    order = order(:);
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
