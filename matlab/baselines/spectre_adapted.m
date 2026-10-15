function order = spectre_adapted(Features, Cost, params, attribs)


    if nargin < 3 || isempty(params), params = struct(); end
    if nargin < 4, attribs = struct(); end
    K       = getf(params,'luK', 10);
    popSize = getf(params,'luPop', 100);
    maxGen  = getf(params,'luGen', 300);
    pc      = getf(params,'luPc', 0.9);
    selLen  = getf(params,'luSelLen', []);
    verbose = getf(params,'luVerbose', 0);

    n = size(Features,1);
    if isempty(selLen), selLen = min(n, 1000); end

    mu = mean(Features,1); sd = std(Features,0,1); sd(sd==0)=1;
    Z  = (Features - mu) ./ sd;

    if isfield(attribs,'diversity') && ~isempty(attribs.diversity)
        diversity = attribs.diversity(:);
    else
        diversity = block_mean_distance(Z, 1000);
    end

    if isfield(attribs,'demand') && ~isempty(attribs.demand)
        demand = attribs.demand(:);
    elseif isfield(attribs,'demandCols') && ~isempty(attribs.demandCols)
        c = Features(:, attribs.demandCols);
        demand = sum(minmax_cols(c), 2);
    else
        qLo = quantile(Features, 0.20, 1);
        qHi = quantile(Features, 0.80, 1);
        demand = sum(Features > qHi | Features < qLo, 2);
    end


    if isfield(attribs,'cp') && ~isempty(attribs.cp)
        cp = attribs.cp(:);
    elseif isfield(attribs,'prevLabels') && ~isempty(attribs.prevLabels)
        cp = knn_unsafe_fraction(Z, logical(attribs.prevLabels(:)), K, 1000);
    else
        cp = minmax_vec(knn_mean_distance(Z, K, 1000));
    end

    comp = (double(cp>0.89) + minmax_vec(demand) + cp + minmax_vec(diversity)) / 4;
    if getf(params,'luCostAware', false)
        comp = comp ./ max(Cost(:), 1e-9);
    end

    L = selLen;
    P = randi(n, popSize, L);
    F = eval_pop_spectre(P, cp, demand, diversity, L);

    for g = 1:maxGen
        rnk = nds_ranks_M(F);
        cd  = crowding_all_M(F, rnk);

        Q = zeros(popSize, L); cnt = 0;
        while cnt < popSize
            i = tour(rnk, cd, popSize);
            j = tour(rnk, cd, popSize);
            if rand < pc
                mask = rand(1,L) < 0.5;
                c1 = P(i,:); c2 = P(j,:);
                tmp = c1(mask); c1(mask) = c2(mask); c2(mask) = tmp;
            else
                c1 = P(i,:); c2 = P(j,:);
            end
            c1 = reset_mutate(c1, 1/L, n);
            c2 = reset_mutate(c2, 1/L, n);
            cnt = cnt+1; Q(cnt,:) = c1;
            if cnt < popSize, cnt = cnt+1; Q(cnt,:) = c2; end
        end
        FQ = eval_pop_spectre(Q, cp, demand, diversity, L);

        R = [P; Q]; FR = [F; FQ];
        rkAll = nds_ranks_M(FR);
        keep  = env_select_M(FR, rkAll, popSize);
        P = R(keep,:); F = FR(keep,:);

        if verbose > 0 && mod(g, verbose) == 0
            fprintf('  [Lu/SPECTRE] gen %d/%d  front1=%d\n', g, maxGen, sum(rkAll(keep)==1));
        end
    end

    rnk = nds_ranks_M(F);
    f1  = find(rnk == 1);
    Ff  = F(f1,:);
    lo = min(Ff,[],1); hi = max(Ff,[],1); rg = hi-lo; rg(rg==0)=1;
    Fn = (Ff - lo) ./ rg;
    [~, kk] = min(sqrt(sum(Fn.^2, 2)));
    seq = P(f1(kk), :);

    [~, ia] = unique(seq, 'stable');
    part = seq(sort(ia));
    rest = setdiff((1:n)', part(:), 'stable');
    [~, ro] = sort(comp(rest), 'descend');
    order = [part(:); rest(ro)];
    order = order(:);
end

function F = eval_pop_spectre(P, cp, demand, diversity, L)
    m = size(P,1); F = zeros(m,5);
    ID = (0:L-1)';
    weight = (L - ID + 1) / L;
    for r = 1:m
        idx = P(r,:)';
        c = cp(idx); dm = demand(idx); dv = diversity(idx);
        collision_number   = sum(c > 0.89);
        potential_number   = sum(c > 0.3 & c <= 0.9);
        high_demand_number = sum(dm >= 6);
        avg_div            = sum(dv) / L;
        wc = sum(weight .* double(c > 0.89));
        wd = sum(weight .* dm);
        wp = sum(weight .* c);
        wv = sum(weight .* dv);
        avg_weight = (wc/L + wd/L + wp/L + wv/L) / 4;
        F(r,:) = -[collision_number, potential_number, high_demand_number, avg_div, avg_weight];
    end
end

function rank = nds_ranks_M(F)
    N = size(F,1); M = size(F,2);
    leq = true(N,N); lt = false(N,N);
    for m = 1:M
        c = F(:,m);
        leq = leq & bsxfun(@le, c, c.');
        lt  = lt  | bsxfun(@lt, c, c.');
    end
    Dom = leq & lt; Dom(1:N+1:end) = false;
    ndom = sum(Dom,1); rank = zeros(1,N); remaining = true(1,N); r = 0;
    while any(remaining)
        r = r + 1;
        cur = remaining & (ndom == 0);
        if ~any(cur), rank(remaining) = r; break; end
        rank(cur) = r; remaining(cur) = false;
        ndom = ndom - sum(Dom(cur,:),1); ndom(cur) = inf;
    end
    rank = rank(:);
end

function cd = crowding_all_M(F, rank)
    N = size(F,1); cd = zeros(N,1);
    for r = 1:max(rank)
        idx = find(rank==r);
        if isempty(idx), continue; end
        cd(idx) = crowd_M(F(idx,:));
    end
end

function d = crowd_M(Fs)
    l = size(Fs,1); d = zeros(l,1);
    if l <= 2, d(:) = inf; return; end
    for m = 1:size(Fs,2)
        [v,o] = sort(Fs(:,m));
        d(o(1)) = inf; d(o(end)) = inf;
        rg = v(end)-v(1);
        if rg <= 0, continue; end
        d(o(2:end-1)) = d(o(2:end-1)) + (v(3:end)-v(1:end-2))/rg;
    end
end

function keep = env_select_M(F, rank, N)
    keep = [];
    for r = 1:max(rank)
        fr = find(rank==r); fr = fr(:);
        if isempty(fr), continue; end
        if numel(keep)+numel(fr) <= N
            keep = [keep; fr];
        else
            c = crowd_M(F(fr,:));
            [~,o] = sort(c,'descend');
            keep = [keep; fr(o(1:N-numel(keep)))];
            break;
        end
    end
    keep = keep(1:N);
end

function w = tour(rank, cd, pop)
    a = randi(pop); b = randi(pop);
    if rank(a) < rank(b), w = a;
    elseif rank(b) < rank(a), w = b;
    elseif cd(a) >= cd(b), w = a; else, w = b; end
end

function c = reset_mutate(c, pm, n)
    hit = rand(1,numel(c)) < pm;
    if any(hit), c(hit) = randi(n, 1, sum(hit)); end
end

function s = block_mean_distance(X, B)
    n = size(X,1); sq = sum(X.^2,2); s = zeros(n,1);
    for a = 1:B:n
        e = min(a+B-1,n); Xb = X(a:e,:);
        d2 = max(sq(a:e) + sq' - 2*(Xb*X'), 0);
        s(a:e) = sum(sqrt(d2), 2);
    end
    s = s / max(n-1,1);
end

function cp = knn_unsafe_fraction(X, prevUnsafe, K, B)
    n = size(X,1); sq = sum(X.^2,2); cp = zeros(n,1);
    for a = 1:B:n
        e = min(a+B-1,n); Xb = X(a:e,:);
        d2 = max(sq(a:e) + sq' - 2*(Xb*X'), 0);
        [~, idx] = sort(d2, 2, 'ascend');
        cp(a:e) = mean(prevUnsafe(idx(:, 2:min(K+1,n))), 2);
    end
end

function y = minmax_vec(x)
    lo = min(x); hi = max(x); r = hi - lo; if r == 0, r = 1; end
    y = (x - lo) / r;
end

function cn = minmax_cols(c)
    lo = min(c,[],1); hi = max(c,[],1); r = hi-lo; r(r==0)=1;
    cn = (c - lo) ./ r;
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end


function md = knn_mean_distance(X, K, B)
    n = size(X,1); sq = sum(X.^2,2); md = zeros(n,1);
    K = max(1, min(K, n-1));
    for a = 1:B:n
        e = min(a+B-1,n); Xb = X(a:e,:);
        d2 = max(sq(a:e) + sq' - 2*(Xb*X'), 0);
        ds = sort(sqrt(d2), 2, 'ascend');
        md(a:e) = mean(ds(:, 2:K+1), 2);
    end
end
