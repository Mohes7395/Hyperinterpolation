% Implementation of the Algorithm 2

function [Phi, X, w, N, r] = getCF(K, N, x_ctrs, ker, meas, y, reduced_param, dim)

% initialize params
r = 0;
w_min = 0;
N_max = 1e5;
max_rank = 0;

while (r < K) || (w_min < 0)
    % fprintf('Iterating with r = %2d and w_min = %2.2e N = %4d\n', r, w_min, N)
    % Get the points
    X = haltonseq(N, dim);

    % Construct the product matrix and compute its rank
    Phi = prod_basis(X, x_ctrs, ker, reduced_param);
    r = rank(Phi);

    % If the rank does not improve, decide that K=r    
    if r > max_rank
        max_rank = r;
    elseif r == max_rank
        K = r;
    end

    % Compute the LS solution
    if r == K  
        % [w,res] = BG_solver(Phi, meas, y, 1);
        %     if res < 1e-9
        %         w(w < 0 ) = 0;
        %     end
        % w_min = min(w);  

        [w, ~] = BG_solver(Phi, meas, y, 1);
        w(w < 0) = 0;    % always zero negligible negatives
        w_min = min(w);  % then check positivity

    end

    % Increment N
    N = N + 1;
    if N > N_max
        disp('Too many points')
        break
    end

end

N = N-1;

end