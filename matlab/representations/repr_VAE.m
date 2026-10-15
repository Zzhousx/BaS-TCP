function [Z, info] = repr_VAE(Features, params)

    if nargin < 2 || isempty(params), params = struct(); end
    d      = getf(params,'vaeDim', getf(params,'k1', 10));
    H      = getf(params,'vaeH', 32);
    epochs = getf(params,'vaeEpochs', 600);
    lr     = getf(params,'vaeLr', 0.01);
    beta   = getf(params,'vaeBeta', 1e-3);

    X = double(Features);
    mu0 = mean(X,1); sd0 = std(X,0,1); sd0(sd0==0) = 1;
    Xn = (X - mu0) ./ sd0;

    [net, finalLoss] = train_vae_full(Xn, d, H, epochs, lr, beta);
    Z = tanh(Xn * net.W1 + net.b1) * net.Wm + net.bm;

    info = struct('dim', d, 'finalLoss', finalLoss);
end

function [net, lossOut] = train_vae_full(X, d, H, epochs, lr, beta)
    [n, D] = size(X);
    W1 = randn(D,H)/sqrt(D); b1 = zeros(1,H);
    Wm = randn(H,d)/sqrt(H); bm = zeros(1,d);
    Wl = randn(H,d)/sqrt(H); bl = zeros(1,d);
    W2 = randn(d,H)/sqrt(d); b2 = zeros(1,H);
    W3 = randn(H,D)/sqrt(H); b3 = zeros(1,D);
    lossOut = NaN;

    for ep = 1:epochs
        h  = tanh(X*W1 + b1);
        mu = h*Wm + bm;
        lv = max(min(h*Wl + bl, 6), -6);
        eps = randn(size(mu));
        z  = mu + exp(0.5*lv) .* eps;
        h2 = tanh(z*W2 + b2);
        xr = h2*W3 + b3;

        rec = sum((xr - X).^2, 'all')/n;
        kl  = 0.5*sum(mu.^2 + exp(lv) - 1 - lv, 'all')/n;
        lossOut = rec + beta*kl;

        dxr = 2*(xr - X)/n;
        gW3 = h2'*dxr;  gb3 = sum(dxr,1);
        dh2 = (dxr*W3') .* (1 - h2.^2);
        gW2 = z'*dh2;   gb2 = sum(dh2,1);
        dz  = dh2*W2';

        dmu = dz + beta*mu/n;
        dlv = dz .* (0.5*exp(0.5*lv).*eps) + beta*0.5*(exp(lv)-1)/n;
        gWm = h'*dmu;   gbm = sum(dmu,1);
        gWl = h'*dlv;   gbl = sum(dlv,1);
        dh  = (dmu*Wm' + dlv*Wl') .* (1 - h.^2);
        gW1 = X'*dh;    gb1 = sum(dh,1);

        W1 = W1 - lr*gW1;  b1 = b1 - lr*gb1;
        Wm = Wm - lr*gWm;  bm = bm - lr*gbm;
        Wl = Wl - lr*gWl;  bl = bl - lr*gbl;
        W2 = W2 - lr*gW2;  b2 = b2 - lr*gb2;
        W3 = W3 - lr*gW3;  b3 = b3 - lr*gb3;
    end
    net = struct('W1',W1,'b1',b1,'Wm',Wm,'bm',bm);
end

function v = getf(s, f, dv)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dv; end
end
