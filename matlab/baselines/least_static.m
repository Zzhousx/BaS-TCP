function order = least_static(Features, labels, params, prevLabels)


    if nargin < 3 || isempty(params), params = struct(); end
    if nargin < 4, prevLabels = []; end
    H      = getf(params,'lsH', 32);
    lr     = getf(params,'lsLr', 0.05);
    epochs = getf(params,'lsEpochs', 300);
    K      = getf(params,'lsFolds', 5);
    l2     = getf(params,'lsL2', 1e-4);

    n = size(Features,1);
    mu = mean(Features,1); sd = std(Features,0,1); sd(sd==0)=1;
    X  = (Features - mu) ./ sd;

    if ~isempty(prevLabels)
        y   = double(prevLabels(:) ~= 0);
        net = train_risk_head(X, y, H, lr, epochs, l2);
        risk = predict_risk(X, net);
    else
        y = double(labels(:) ~= 0);
        idx = randperm(n);
        foldOf = zeros(n,1);
        foldOf(idx) = mod(0:n-1, K) + 1;
        risk = zeros(n,1);
        for k = 1:K
            te = (foldOf == k); tr = ~te;
            net = train_risk_head(X(tr,:), y(tr), H, lr, epochs, l2);
            risk(te) = predict_risk(X(te,:), net);
        end
    end

    [~, order] = sort(risk, 'descend');
    order = order(:);
end

function net = train_risk_head(X, y, H, lr, epochs, l2)
    [n, d] = size(X);
    W1 = randn(d,H)/sqrt(d); b1 = zeros(1,H);
    W2 = randn(H,1)/sqrt(H); b2 = 0;

    pPos = max(mean(y), 1e-6);
    wPos = 0.5/pPos; wNeg = 0.5/max(1-pPos, 1e-6);
    w = y*wPos + (1-y)*wNeg;

    for ep = 1:epochs
        Z1 = X*W1 + b1;  A1 = tanh(Z1);
        z  = A1*W2 + b2; p = 1 ./ (1 + exp(-z));

        g   = w .* (p - y) / n;
        gW2 = A1'*g + l2*W2;  gb2 = sum(g);
        gA1 = g*W2';  gZ1 = gA1 .* (1 - A1.^2);
        gW1 = X'*gZ1 + l2*W1; gb1 = sum(gZ1,1);

        W2 = W2 - lr*gW2;  b2 = b2 - lr*gb2;
        W1 = W1 - lr*gW1;  b1 = b1 - lr*gb1;
    end
    net = struct('W1',W1,'b1',b1,'W2',W2,'b2',b2);
end

function risk = predict_risk(X, net)
    z = tanh(X*net.W1 + net.b1)*net.W2 + net.b2;
    risk = 1 ./ (1 + exp(-z));
end

function v = getf(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
