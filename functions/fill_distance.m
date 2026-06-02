function h = fill_distance(x_ctrs, n_fine)
    
dim = size(x_ctrs, 2);
    
    if dim ==1
        
        X_fine = get_ctrs(n_fine, "Uniform", dim);

    elseif dim ==2
        if n_fine >= 10^3
            n_fine = 1e03;
        end
        X_fine = get_ctrs(n_fine, "Uniform", dim);

    end

% For each fine grid point, find distance to nearest data site
% pdist2 returns (n_fine x n_ctrs) matrix of pairwise distances
D = pdist2(X_fine, x_ctrs);          % n_fine x n
min_dist = min(D, [], 2);            % nearest neighbour distance per fine point

% Fill distance = maximum over all fine grid points
h = max(min_dist);


end