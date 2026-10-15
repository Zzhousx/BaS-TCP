function [Z, info] = repr_ISA(Features, labels, params)


    if nargin < 3 || isempty(params), params = struct(); end
    d     = 10;
    reg   = getf(params,'isaReg', 1e-6);
    doStd = getf(params,'isaStandardize', true);

    X = double(Features);
    y = double(labels(:) ~= 0);
    [n, D] = size(X);

    if doStd
        mu = mean(X,1); sd = std(X,0,1); sd(sd==0) = 1;
        X = (X - mu) ./ sd;
    end
    cls = unique(y);
    gm  = mean(X,1);
    Sb  = zeros(D,D); Sw = zeros(D,D);
    for c = cls'
        Xc = X(y==c, :);
        if isempty(Xc), continue; end
        mc = mean(Xc,1); nc = size(Xc,1);
        dm = (mc - gm)';
        Sb = Sb + nc * (dm * dm');
        Zc = Xc - mc;
        Sw = Sw + (Zc' * Zc);
    end
    Sw = Sw + reg * trace(Sw)/D * eye(D);

    [V, Dg] = eig(Sb, Sw);
    [ev, idx] = sort(real(diag(Dg)), 'descend');
    V = real(V(:, idx));

    nDisc = min(d, numel(cls)-1);
    W = V(:, 1:nDisc);

    if nDisc < d
        P = eye(D) - W * ((W'*W) \ W');
        Xr = X * P;
        [Vr, ~] = eig(cov(Xr));
        [~, ir] = sort(diag(cov(Xr)), 'descend'); %#ok<*ASGLU>
        [Cr, Lr] = eig(cov(Xr));
        [~, ord] = sort(real(diag(Lr)), 'descend');
        W = [W, real(Cr(:, ord(1:d-nDisc)))];
    end

    Z = X * W;
    info = struct('dim', size(Z,2), 'nDiscriminant', nDisc, ...
                  'eigenvalues', ev(1:min(d, numel(ev))));
end

function v = getf(s, f, dv)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dv; end
end
