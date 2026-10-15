function [labels, centroids] = density_kmeans(M, c)
    if c ~= 2
        error('This density_kmeans is designed only for c=2.');
    end
    [n, dim] = size(M);
    max_iter = 100;
    K_knn = min(5, n-1);
    [~, dists] = knnsearch(M, M, 'K', K_knn + 1);
    density_metric = dists(:, end);

    [~, majority_idx] = min(density_metric); % Densest point
    dist_to_c1 = pdist2(M, M(majority_idx,:));
    objective_for_c2 = dist_to_c1 .* (density_metric.^0.5);
    [~, minority_idx] = max(objective_for_c2); % Far and sparse point

    centroids = [M(majority_idx, :); M(minority_idx, :)];
    labels = zeros(n, 1);
    for i = 1:max_iter
        D = pdist2(M, centroids); % n x 2 matrix of distances



        [~, new_labels] = min(D, [], 2);

        if isequal(labels, new_labels)
            break;
        end
        labels = new_labels;

        centroids(1,:) = mean(M(labels==1, :), 1);
        centroids(2,:) = mean(M(labels==2, :), 1);

        if any(isnan(centroids(:)))
             [~, majority_idx] = min(density_metric);
             dist_to_c1 = pdist2(M, M(majority_idx,:));
             objective_for_c2 = dist_to_c1 .* (density_metric.^0.5);
             [~, minority_idx] = max(objective_for_c2);
             centroids = [M(majority_idx, :); M(minority_idx, :)];
             warning('Empty cluster detected during iteration, re-initializing centroids for this step.');
        end
    end
end
