function y = moment(ker, x_ctrs, reduced_param)
    % Add explicit / symbolic computations when possible

    L = 1e5;
    X = haltonseq(L, size(x_ctrs, 2)); 
    K = prod_basis(X, x_ctrs, ker, reduced_param);

    y = sum(K, 2) / L;

end