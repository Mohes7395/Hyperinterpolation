function K = prod_basis(X, x_ctrs, ker, reduced_param)
    
    m = size(X, 1);
    n = size(x_ctrs, 1);
    A = ker(X, x_ctrs);
    
    if reduced_param == 1
        A = [ones(m, 1), A];
        K = A;
    else
        K = [];
        for i = 1:n
            for j = i:n
                M = A(:,i).*A(:,j);
                K = [K, M];
            end
        end
        K = [ones(m, 1), A, K];
    end

    K = K';

end