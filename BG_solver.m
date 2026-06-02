% The Backus Gilbert implementation, use to solve eq (13)

function[w, residual] = BG_solver(Phi, meas, y, stability_param)

% Option 1: It was previously written by me and Gabrielr, but I doubt
% if it is correct.
%Rs_sqrt = sqrt(meas / M) * eye(M);
%w = Rs_sqrt * pinv(Phi * Rs_sqrt) * y;

M = size(Phi,2);
Rs = (meas / M) * eye(M);
ARAT = Phi * Rs * Phi';

if stability_param == 1
    % Regularization
    reg = 1e-10 * trace(ARAT) / size(ARAT,1);

    % Solve
    lambda = (ARAT + reg * eye(size(ARAT))) \ y;

    % Recover weights
    w = Rs * Phi' * lambda;

    % Residual
    residual = norm(Phi*w - y) / norm(y);
else
    %Solve (A R A^T) λ = b
    lambda = ARAT \ y;
    %Recover weights
    w = Rs * Phi' * lambda;
    residual = 0;
end


end