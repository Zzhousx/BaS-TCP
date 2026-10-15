function order = rl_static(Features, Cost, labels, params, prevLabels)


    if nargin < 4 || isempty(params), params = struct(); end
    if nargin < 5, prevLabels = []; end
    H    = getf(params,'rlH',32);
    lr   = getf(params,'rlLr',0.05);
    E    = getf(params,'rlEpisodes',400);
    B    = getf(params,'rlBatch',256);
    Sopp = getf(params,'rlSampleOpp',40);
    K    = getf(params,'rlFolds',5);
    proto= getf(params,'rlProtocol','auto');

    n = size(Features,1);
    Cost = Cost(:);

    mu = mean(Features,1); sd = std(Features,0,1); sd(sd==0)=1;
    X  = (Features - mu) ./ sd;

    if strcmpi(proto,'auto')
        if ~isempty(prevLabels), proto = 'prev'; else, proto = 'crossfit'; end
    end

    if strcmpi(proto,'prev')
        if isempty(prevLabels), error('protocol=prev needs prevLabels。'); end
        y = double(prevLabels(:) ~= 0);
        net = train_pairwise_dqn(X, y, Cost, H, lr, E, B);
        order = rank_by_pairwise(X, net, Sopp);
    else
        y = double(labels(:) ~= 0);
        fold = mod(0:n-1, K) + 1;
        idx = randperm(n); foldOf = zeros(n,1); foldOf(idx) = fold;
        score = zeros(n,1);
        for k = 1:K
            te = (foldOf == k); tr = ~te;
            net = train_pairwise_dqn(X(tr,:), y(tr), Cost(tr), H, lr, E, B);
            score(te) = pairwise_score(X(te,:), X(tr,:), net, Sopp);
        end
        [~, order] = sort(score, 'descend');
    end
    order = order(:);
end

function net = train_pairwise_dqn(X, y, cost, H, lr, E, B)
    [n, d] = size(X);
    din = 2*d;
    W1 = randn(din,H)/sqrt(din); b1 = zeros(1,H);
    W2 = randn(H,1)/sqrt(H);     b2 = 0;
    for ep = 1:E
        i = randi(n, B, 1);  j = randi(n, B, 1);
        pf = [X(i,:), X(j,:)];                 % B×2d

        fi = y(i); fj = y(j); ci = cost(i); cj = cost(j);
        pref = 0.5*ones(B,1);
        pref(fi > fj) = 1;  pref(fj > fi) = 0;
        tie = (fi == fj);
        pref(tie & (ci < cj)) = 1;
        pref(tie & (ci > cj)) = 0;

        Z1 = pf*W1 + b1; A1 = tanh(Z1);
        q  = A1*W2 + b2;
        pred = 1 ./ (1 + exp(-q));

        g  = (pred - pref) / B;
        gW2 = A1'*g;  gb2 = sum(g);
        gA1 = g*W2';  gZ1 = gA1 .* (1 - A1.^2);
        gW1 = pf'*gZ1; gb1 = sum(gZ1,1);

        W2 = W2 - lr*gW2;  b2 = b2 - lr*gb2;
        W1 = W1 - lr*gW1;  b1 = b1 - lr*gb1;
    end
    net = struct('W1',W1,'b1',b1,'W2',W2,'b2',b2);
end

function order = rank_by_pairwise(X, net, Sopp)
    score = pairwise_score(X, X, net, Sopp);
    [~, order] = sort(score, 'descend');
end

function score = pairwise_score(Xq, Xref, net, Sopp)
    nq = size(Xq,1); nr = size(Xref,1);
    Sopp = min(Sopp, nr);
    score = zeros(nq,1);
    for a = 1:nq
        opp = randi(nr, Sopp, 1);
        pf  = [repmat(Xq(a,:), Sopp, 1), Xref(opp,:)];
        q   = tanh(pf*net.W1 + net.b1)*net.W2 + net.b2;
        score(a) = mean(1 ./ (1 + exp(-q)));
    end
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
