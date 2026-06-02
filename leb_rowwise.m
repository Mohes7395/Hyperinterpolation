function val = leb_rowwise(K_leb, VtWV, VZ, W)
% Precomputes B = (VtWV)^{-1} * VZ' * W  (n x M) once,
% then evaluates one row of H = K_leb * B at a time.
    n_leb = size(K_leb, 1);
    B     = VtWV \ (VZ' * W);    % n x M — computed once, reused
    val   = 0;
    for i = 1:n_leb
        h_i = K_leb(i,:) * B;   % 1 x M
        val = max(val, sum(abs(h_i)));
    end
end