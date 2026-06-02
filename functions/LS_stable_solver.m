% TO solve eq (10)

function a = LS_stable_solver(W,V,y, param)
 
V_tild = sqrt(W)*V;
y_tild = sqrt(W)*y;

if param == "QR"
    [Q,R] = qr(V_tild);
    a = R \ (Q'*y_tild);
elseif param == "SVD"
    [U,S,Vsvd] = svd(V_tild,'econ');
    s = diag(S);

    tol = 1e-12 * max(s);
    r = sum(s > tol);
    a = Vsvd(:,1:r) * ((U(:,1:r)' * y_tild) ./ s(1:r));
end

end