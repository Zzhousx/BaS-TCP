function sorted_indices = balanced_prioritize(Z, costs, k3, k4)
num_tests = size(Z, 1);
if nargin < 2 || isempty(costs)
    costs = ones(num_tests, 1);
end
eta = k3;  beta = k4;  epsilon = 1e-9;

mu = mean(Z, 1);  sg = std(Z, 0, 1);  sg(sg == 0) = 1;
Zn = (Z - mu) ./ sg;
sqn = sum(Zn.^2, 2);

risk_proxies = sqrt(sqn);
min_r = min(risk_proxies);  max_r = max(risk_proxies);
if (max_r - min_r) > epsilon
    risk_proxies = (risk_proxies - min_r) / (max_r - min_r);
else
    risk_proxies = zeros(num_tests, 1);
end
rw = 1 + eta * risk_proxies;

distcol = @(u) sqrt(max(sqn + sqn(u) - 2*(Zn * Zn(u,:)'), 0));

sum_all = zeros(num_tests, 1);
best = -inf;  ra = 1;  rb = 2;
for p = 1:num_tests
    d = distcol(p);
    sum_all = sum_all + d;
    d(p) = -inf;
    [mx, idx] = max(d);
    if mx > best, best = mx; ra = p; rb = idx; end
end
s1 = max(ra, rb);  s2 = min(ra, rb);
d1 = distcol(s1);  d2 = distcol(s2);

sum_pool = sum_all - d1 - d2;
wsum_sel = d1*rw(s1) + d2*rw(s2);

permutation = zeros(num_tests, 1);
permutation(1) = s1;  permutation(2) = s2;
unselected_mask = true(num_tests, 1);
unselected_mask([s1 s2]) = false;

for i = 3:num_tests
    m = num_tests - (i-1);
    future_values = 1 ./ (sum_pool ./ max(m-1, 1) + epsilon);
    immediate_rewards = wsum_sel ./ (costs + epsilon);

    cand = unselected_mask;
    ir = immediate_rewards(cand);
    fv = future_values(cand);
    if std(ir) > epsilon, ir = (ir - mean(ir)) / std(ir); end
    if std(fv) > epsilon, fv = (fv - mean(fv)) / std(fv); end

    q_values = ir + beta * fv;
    idxc = find(cand);
    [~, best_local] = max(q_values);
    u = idxc(best_local);

    permutation(i) = u;
    unselected_mask(u) = false;
    du = distcol(u);
    sum_pool = sum_pool - du;
    wsum_sel = wsum_sel + du * rw(u);
end

sorted_indices = permutation;
end
