function order = art_tcp(Features, candPool, Cost, costAware)

    if nargin < 2 || isempty(candPool), candPool = 10; end
    if nargin < 3 || isempty(Cost), Cost = ones(size(Features,1),1); end
    if nargin < 4 || isempty(costAware), costAware = false; end

    n = size(Features,1);
    Cost = Cost(:);

    mu = mean(Features,1); sd = std(Features,0,1); sd(sd==0)=1;
    X  = (Features - mu) ./ sd;

    order     = zeros(n,1);
    remaining = true(n,1);
    minDist   = inf(n,1);

    first = randi(n);
    order(1) = first; remaining(first) = false;
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
            score = minDist(pool) ./ max(Cost(pool), 1e-9);
        else
            score = minDist(pool);
        end
        [~, j] = max(score);
        best = pool(j);

        order(t) = best; remaining(best) = false;
        d = sqrt(sum((X - X(best,:)).^2, 2));
        minDist = min(minDist, d); minDist(best) = -inf;
    end
    order = order(:);
end
