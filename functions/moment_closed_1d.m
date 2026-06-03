function b = moment_closed_1d(rbf_name, x_ctrs, ep)
% Closed-form moments on Omega = [0,1] for reduced (CFv) exactness.
%
% Reduced exactness is imposed on (N_K(X) + 1) x 1, so the RHS is
%   b = [ b_1, ..., b_n, b_0 ]^T  with the SAME ordering produced by
%   prod_basis(..., reduced_param=1), namely  K = [ ones, A ]  ->  K' has
%   the constant row FIRST.  We therefore return b = [ b_const ; b_kernels ].
%
%   b_const   = \int_0^1 1 dx                       = 1
%   b_i       = \int_0^1 phi(ep |x - x_i|) dx       = (1/ep)[ Phi(ep x_i) + Phi(ep (1-x_i)) ]
%
% where Phi(s) = \int_0^s phi(t) dt is the radial antiderivative:
%   LMatern : phi(t) = (1+t) e^{-t}        ->  Phi(s) = 2 - (s+2) e^{-s}
%   QMatern : phi(t) = (3+3t+t^2) e^{-t}   ->  Phi(s) = 8 - (s^2+5s+8) e^{-s}
%             (matches select_rbf.m, which uses the *3-scaled QMatern)
%
% Inputs:
%   rbf_name : "LMatern" or "QMatern"
%   x_ctrs   : n x 1 centers in [0,1]
%   ep       : shape parameter (scalar)
%
% Output:
%   b : (n+1) x 1 moment vector, ordered [const ; kernels] to match
%       prod_basis(X, x_ctrs, ker, 1)'  used in getCF.

    x_ctrs = x_ctrs(:);
    n = numel(x_ctrs);

    switch rbf_name
        case "LMatern"
            Phi = @(s) 2 - (s + 2).*exp(-s);
        case "QMatern"
            Phi = @(s) 8 - (s.^2 + 5*s + 8).*exp(-s);
        otherwise
            error('moment_closed_1d: kernel %s not supported (use LMatern or QMatern).', rbf_name);
    end

    s_left  = ep * x_ctrs;          % distance ep*(x_i - 0)
    s_right = ep * (1 - x_ctrs);    % distance ep*(1 - x_i)

    b_kernels = (Phi(s_left) + Phi(s_right)) / ep;   % n x 1
    b_const   = 1;                                   % \int_0^1 1 dx

    b = [b_const; b_kernels];       % (n+1) x 1, constant first
end
