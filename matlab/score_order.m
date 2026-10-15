function score=score_order(order,y,costs)
order=order(:);y=double(y(:));costs=costs(:);n=numel(y);m=sum(y);
assert(isequal(sort(order),(1:n)'));
if all(costs==1)
 score=1-sum(find(y(order)))/(n*m)+1/(2*n);
else
 score=trapz(cumsum(costs(order)),cumsum(y(order)))/(sum(costs)*m);
end
end
