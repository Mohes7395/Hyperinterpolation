function val = leb_svd(K_leb, VZ, W, tol_rel)
% SVD-consistent Lebesgue constant (ell^inf(Z) -> Linf(Omega) operator norm).
%
% Builds the hat matrix H = K_leb * Vtrunc^+  through the SAME truncated-SVD
% pseudoinverse of sqrt(W)*VZ that LS_stable_solver uses for the coefficients,
% instead of inverting the squared normal matrix VZ'*W*VZ. This removes the
% kappa(VZ)^2 amplification that otherwise contaminates the Lebesgue estimate.
%
% Inputs:
%   K_leb : n_leb x n   kernel translates evaluated on the Lebesgue grid
%   VZ    : M x n        kernel translates evaluated at cubature nodes Z
%   W     : M x M        diagonal cubature-weight matrix
%   tol_rel (optional)   relative SVD truncation (default 1e-12, matches solver)
%
% Output:
%   val   : Lebesgue constant = max_x sum_k |H(x,k)|
%
% Derivation:
%   coefficients solve  min_a || sqrt(W)(VZ a - y) ||_2
%   => a = Vt^+ (sqrt(W) y),  Vt = sqrt(W) VZ
%   approximant on grid = K_leb a = [K_leb Vt^+ sqrt(W)] y
%   so the data-to-approximant map is H = K_leb * Vt^+ * sqrt(W).

    if nargin < 4 || isempty(tol_rel)
        tol_rel = 1e-12;
    end

    sW = sqrt(W);
    Vt = sW * VZ;                 % M x n  (weighted design matrix)

    [U, S, Vsvd] = svd(Vt, 'econ');
    s   = diag(S);
    tol = tol_rel * max(s);
    r   = sum(s > tol);

    % Pinv applied to sqrt(W): B = Vt^+ * sqrt(W)  (n x M)
    %   Vt^+ = Vsvd(:,1:r) * diag(1./s(1:r)) * U(:,1:r)'
    B = Vsvd(:,1:r) * ((U(:,1:r)' * sW) ./ s(1:r));   % n x M

    H = K_leb * B;                % n_leb x M
    val = max(sum(abs(H), 2));
end
