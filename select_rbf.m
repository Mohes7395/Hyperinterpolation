function rbf = select_rbf(rbf_name)

    if rbf_name == "Gaussian"
        rbf = @(r, ep) exp(-ep*r.^2);
    elseif rbf_name == "BasicMatern"
        rbf = @(r, ep) exp(- ep * r);
    elseif rbf_name == "LMatern"
        rbf = @(r, ep) (1 + ep*r).*exp(-ep * r);
    elseif rbf_name == "QMatern"
        rbf = @(r, ep) (3 + 3*ep*r + (ep*r).^2).*exp(-ep * r);
    elseif rbf_name == "Wendland"
        rbf = @(r, ep) max(0, (1 - ep * r)).^4 .* (4*ep*r + 1);
    end

end
