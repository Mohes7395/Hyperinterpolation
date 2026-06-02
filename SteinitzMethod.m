%Input : 1) the matrix Phi contains evaluation of basis in CF points
% 2) K is the number of kernels' product
% 3) N is the initial number of CF points
% 4) w is is the initial CF points
% 5) X is the initial CF points

function [Phi, N, w, X] = SteinitzMethod(Phi, K, N, w, X)

while K < N
    
    nullSpace = null(Phi);
    a = nullSpace(:,1);
    sigma = max(a./w);
    ww = (sigma * w - a)/sigma;
    w = ww;
    zeroIndices = find(w == 0);
    w(zeroIndices) = [];
    X(zeroIndices,:) = [];
    Phi(:,zeroIndices') = [];

    N = N - length(zeroIndices);

end
    
end
