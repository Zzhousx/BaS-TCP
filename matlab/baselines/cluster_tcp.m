function [order, info] = cluster_tcp(Features, Cost, params)


    if nargin < 3 || isempty(params), params = struct(); end
    n = size(Features,1);
    Cost = Cost(:);

    KMode    = getf(params,'clusterKMode','auto');
    Kfixed   = getf(params,'clusterK', []);
    KRange   = getf(params,'clusterKRange', []);
    algo     = getf(params,'clusterAlgo','kmeans');
    intra    = getf(params,'clusterIntra','art');
    restarts = getf(params,'clusterRestarts', 5);
    maxit    = getf(params,'clusterMaxIt', 100);
    costAware= getf(params,'clusterCostAware', false);
    silCap   = getf(params,'clusterSilSample', 3000);
    fcmM     = getf(params,'clusterFcmM', 2.0);
    artPool  = getf(params,'clusterArtPool', 10);

    mu = mean(Features,1); sd = std(Features,0,1); sd(sd==0)=1;
    Z  = (Features - mu) ./ sd;

    gc   = mean(Z,1);
    risk = sqrt(sum((Z - gc).^2, 2));

    if isempty(KRange)
        Kmax   = max(2, min(20, floor(n/5)));
        KRange = 2:Kmax;
    end

    silCurve = [];
    if strcmpi(KMode,'fixed') && ~isempty(Kfixed)
        bestK = min(Kfixed, n);
        lbl   = run_clustering(Z, bestK, algo, restarts, maxit, fcmM);
    else
        if n > silCap, sidx = randperm(n, silCap); else, sidx = 1:n; end
        Zs = Z(sidx, :);
        bestSil = -inf; bestK = KRange(1); lbl = [];
        for K = KRange
            if K >= n, continue; end
            lblK = run_clustering(Z, K, algo, restarts, maxit, fcmM);
            s = silhouette_score(Zs, lblK(sidx));
            silCurve(end+1,:) = [K, s]; %#ok<AGROW>
            if s > bestSil
                bestSil = s; bestK = K; lbl = lblK;
            end
        end
        if isempty(lbl)
            bestK = 2; lbl = run_clustering(Z, 2, algo, restarts, maxit, fcmM);
        end
    end
    present = unique(lbl(:))';
    Kp = numel(present);
    queues = cell(Kp,1);
    clusterRisk = zeros(Kp,1);

    if costAware
        key = risk ./ max(Cost, 1e-9);
    else
        key = risk;
    end

    for t = 1:Kp
        members = find(lbl == present(t));
        clusterRisk(t) = mean(risk(members));
        if strcmpi(intra,'art') && numel(members) > 2
            queues{t} = members(art_sequence(Z(members,:), artPool, ...
                                              Cost(members), costAware));
        else
            [~, o] = sort(key(members), 'descend');
            queues{t} = members(o);
        end
    end
    [~, visitOrder] = sort(clusterRisk, 'descend');
    head = ones(Kp,1);
    len  = cellfun(@numel, queues);
    order = zeros(n,1); w = 0; remaining = sum(len);
    while remaining > 0
        for t = visitOrder(:)'
            if head(t) <= len(t)
                w = w + 1;
                order(w) = queues{t}(head(t));
                head(t) = head(t) + 1;
                remaining = remaining - 1;
            end
        end
    end
    order = order(:);

    info = struct('K', numel(present), 'chosenK', bestK, ...
                  'silhouetteCurve', silCurve, 'algo', algo, 'intra', intra);
end

function lbl = run_clustering(X, K, algo, restarts, maxit, fcmM)
    if strcmpi(algo,'fcm')
        lbl = fcm_cluster(X, K, restarts, maxit, fcmM);
    else
        lbl = kmeans_pp(X, K, restarts, maxit);
    end
end

function bestLbl = kmeans_pp(X, K, restarts, maxit)
    [n, d] = size(X); sqX = sum(X.^2,2);
    bestInertia = inf; bestLbl = ones(n,1);
    for rs = 1:restarts
        C = zeros(K,d);
        C(1,:) = X(randi(n),:);
        d2 = max(sqX + sum(C(1,:).^2) - 2*(X*C(1,:)'), 0);
        for c = 2:K
            s = sum(d2);
            if s <= 0, idx = randi(n); else, idx = weighted_sample(d2/s); end
            C(c,:) = X(idx,:);
            d2 = min(d2, max(sqX + sum(C(c,:).^2) - 2*(X*C(c,:)'), 0));
        end
        lbl = zeros(n,1);
        for it = 1:maxit
            D2 = sqX + sum(C.^2,2)' - 2*(X*C');
            [~, newLbl] = min(D2, [], 2);
            if isequal(newLbl, lbl), break; end
            lbl = newLbl;
            for k = 1:K
                mask = (lbl==k);
                if any(mask)
                    C(k,:) = mean(X(mask,:),1);
                else
                    Dtmp = sqX + sum(C.^2,2)' - 2*(X*C');
                    [~, far] = max(min(Dtmp,[],2));
                    C(k,:) = X(far,:);
                end
            end
        end
        inertia = sum(sum((X - C(lbl,:)).^2,2));
        if inertia < bestInertia, bestInertia = inertia; bestLbl = lbl; end
    end
end

function lbl = fcm_cluster(X, K, restarts, maxit, m)
    [n, d] = size(X);
    bestObj = inf; lbl = ones(n,1);
    for rs = 1:restarts
        U = rand(n, K); U = U ./ sum(U,2);
        prevObj = inf; Ubest = U;
        for it = 1:maxit
            Um = U.^m;
            C  = (Um' * X) ./ max(sum(Um,1)', eps);
            D2 = max(sum(X.^2,2) + sum(C.^2,2)' - 2*(X*C'), 1e-12);
            invD = D2 .^ (-1/(m-1));
            U = invD ./ sum(invD, 2);
            obj = sum(sum((U.^m) .* D2));
            if abs(prevObj - obj) < 1e-6 * max(1,abs(prevObj))
                prevObj = obj; Ubest = U; break;
            end
            prevObj = obj; Ubest = U;
        end
        if prevObj < bestObj
            bestObj = prevObj;
            [~, lbl] = max(Ubest, [], 2);
        end
    end
end

function s = silhouette_score(X, lbl)
    n = size(X,1);
    ks = unique(lbl(:))';
    if numel(ks) < 2, s = -1; return; end
    D = squareform(pdist(X));
    a = zeros(n,1); b = inf(n,1);
    for k = ks
        mk = (lbl == k);
        nk = sum(mk);
        sumIn = sum(D(:, mk), 2);
        a(mk) = sumIn(mk) ./ max(nk - 1, 1);
        other = ~mk;
        if nk > 0
            meanToK = sumIn ./ max(nk,1);
            b(other) = min(b(other), meanToK(other));
        end
    end
    denom = max(a, b); denom(denom == 0) = 1;
    sil = (b - a) ./ denom;
    sil(isnan(sil) | isinf(sil)) = 0;
    s = mean(sil);
end

function ordLocal = art_sequence(X, candPool, cst, costAware)
    if nargin < 3 || isempty(cst), cst = ones(size(X,1),1); end
    if nargin < 4 || isempty(costAware), costAware = false; end
    n = size(X,1);
    ordLocal = zeros(n,1);
    remaining = true(n,1);
    minDist = inf(n,1);
    first = randi(n);
    ordLocal(1) = first; remaining(first) = false;
    d = sqrt(sum((X - X(first,:)).^2, 2));
    minDist = min(minDist, d); minDist(first) = -inf;
    for t = 2:n
        remIdx = find(remaining);
        if numel(remIdx) <= candPool
            pool = remIdx;
        else
            pool = remIdx(randperm(numel(remIdx), candPool));
        end
        if costAware
            sc = minDist(pool) ./ max(cst(pool), 1e-9);
        else
            sc = minDist(pool);
        end
        [~, j] = max(sc);
        best = pool(j);
        ordLocal(t) = best; remaining(best) = false;
        d = sqrt(sum((X - X(best,:)).^2, 2));
        minDist = min(minDist, d); minDist(best) = -inf;
    end
end

function idx = weighted_sample(p)
    c = cumsum(p); r = rand;
    idx = find(c >= r, 1, 'first');
    if isempty(idx), idx = numel(p); end
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
