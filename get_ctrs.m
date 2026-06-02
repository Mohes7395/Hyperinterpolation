function x_ctrs = get_ctrs(n_ctrs, dist, dim)
    
    if dim == 1
        if dist == "Halton"
            x_ctrs = haltonseq(n_ctrs, dim);
        elseif dist == "Uniform"
            x_ctrs = linspace(0,1,n_ctrs)';
        end
    elseif dim == 2
        if dist == "Halton"
            x_ctrs = haltonseq(n_ctrs, dim);
        elseif dist == "Uniform"
           [X,Y] = meshgrid(linspace(0,1,n_ctrs));
           x_ctrs = [X(:), Y(:)];
        end
    end

end