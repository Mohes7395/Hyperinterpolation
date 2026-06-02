clear; clc; close all

%% =========================================================================
%  TEST 4 — RUNGE FUNCTION  (dim = 1)   [Lebesgue computation corrected]
%
%  Lebesgue constants for ALL projection methods (polynomial hyper, kernel
%  Hyper Full, kernel Hyper Reduced) computed through the SAME truncated-SVD
%  pseudoinverse used for the coefficients (leb_svd.m), not via the squared
%  normal matrix VtWV. Removes kappa^2 contamination in the kernel methods.
%
%  All other settings unchanged.
% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 1;
data_distribution = "Uniform";
solver            = "SVD";

f = @(x) 1 ./ (1 + 25 * (2*x - 1).^2);

rbf_names = ["BasicMatern", "LMatern", "QMatern"];
tau_vec   = [1, 2, 3];

C_ep_vec   = [300, 500, 700];

n_ctrsVec = [20, 30, 45, 60, 80, 105, 140, 180, 200];
anchor_idx = 2;

q_max      = 50;
q_vec      = 1:3:q_max;
n_poly_vec = q_vec + 1;

n_test = 500;
X_test = linspace(0, 1, n_test)';
f_test = f(X_test);

n_leb = 1000;
X_leb = linspace(0, 1, n_leb)';

%% -------------------------------------------------------------------------
%  STORAGE
% -------------------------------------------------------------------------
nk = length(rbf_names);
nn = length(n_ctrsVec);
nq = length(q_vec);

rms_hyper_full  = zeros(nk, nn);
rms_hyper_red   = zeros(nk, nn);
cond_hyper_full = zeros(nk, nn);
cond_hyper_red  = zeros(nk, nn);
leb_hyper_full  = zeros(nk, nn);
leb_hyper_red   = zeros(nk, nn);
ep_used         = zeros(nk, nn);

rms_poly  = zeros(1, nq);
cond_poly = zeros(1, nq);
leb_poly  = zeros(1, nq);

%% -------------------------------------------------------------------------
%  POLYNOMIAL HYPERINTERPOLATION
% -------------------------------------------------------------------------
fprintf('=== Polynomial Hyperinterpolation (q = 1:3:%2d) ===\n', q_max);
for qi = 1:nq
    q   = q_vec(qi);
    n_p = q + 1;
    M_p = 2 * n_p;

    try
        [z_gl, w_gl] = lgwt(M_p, 0, 1);
    catch
        z_gl = linspace(0, 1, M_p)';
        w_gl = ones(M_p, 1) / M_p;
    end

    V_gl = zeros(M_p, n_p);
    for j = 1:n_p
        V_gl(:,j) = sqrt(2*(j-1)+1) * legendreP(j-1, 2*z_gl-1);
    end

    f_gl   = f(z_gl);
    coeffs = V_gl' * (w_gl .* f_gl);

    V_test = zeros(n_test, n_p);
    for j = 1:n_p
        V_test(:,j) = sqrt(2*(j-1)+1) * legendreP(j-1, 2*X_test-1);
    end
    rms_poly(qi) = rmse(V_test * coeffs, f_test);

    W_gl          = diag(w_gl);
    cond_poly(qi) = cond(sqrt(W_gl) * V_gl);

    V_leb = zeros(n_leb, n_p);
    for j = 1:n_p
        V_leb(:,j) = sqrt(2*(j-1)+1) * legendreP(j-1, 2*X_leb-1);
    end
    % SVD-consistent Lebesgue
    leb_poly(qi) = leb_svd(V_leb, V_gl, W_gl, 1e-12);

    fprintf('  q = %2d  (n_poly=%2d)  |  RMSE=%.3e  Cond=%.3e  Leb=%.3e\n', ...
        q, n_p, rms_poly(qi), cond_poly(qi), leb_poly(qi));
end

%% -------------------------------------------------------------------------
%  KERNEL HYPERINTERPOLATION
% -------------------------------------------------------------------------
for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);
    C_ep     = C_ep_vec(k);

    fprintf('\n=== Kernel: %s | C_ep=%.2f ===\n', rbf_name, C_ep);

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);

        h_X            = fill_distance(x_ctrs, 1000);
        ep             = C_ep * h_X;
        ep_used(k,idx) = ep;
        ker            = @(x, y) rbf(pdist2(x, y), ep);

        fprintf('  n = %3d  |  ep = %7.2f\n', n_ctrs, ep);

        K_leb = ker(X_leb, x_ctrs);

        % (A) Full exactness
        reduced_param = 0;
        y        = moment(ker, x_ctrs, reduced_param);
        r_target = size(y, 1);
        N_init   = 6 * (n_ctrs + 1);
        meas     = 1;

        [~, Z_full, w_full, ~, ~] = getCF(r_target, N_init, x_ctrs, ...
            ker, meas, y, reduced_param, dim);

        VZ_full = ker(Z_full, x_ctrs);
        W_full  = diag(w_full);
        a_full  = LS_stable_solver(W_full, VZ_full, f(Z_full), solver);

        rms_hyper_full(k,idx)  = rmse(ker(X_test, x_ctrs) * a_full, f_test);
        cond_hyper_full(k,idx) = cond(sqrt(W_full) * VZ_full);
        leb_hyper_full(k,idx)  = leb_svd(K_leb, VZ_full, W_full, 1e-12);

        % (B) Reduced exactness
        reduced_param = 1;
        y        = moment(ker, x_ctrs, reduced_param);
        r_target = size(y, 1);
        N_init   = 2 * (n_ctrs + 1);

        [~, Z_red, w_red, ~, ~] = getCF(r_target, N_init, x_ctrs, ...
            ker, meas, y, reduced_param, dim);

        VZ_red = ker(Z_red, x_ctrs);
        W_red  = diag(w_red);
        a_red  = LS_stable_solver(W_red, VZ_red, f(Z_red), solver);

        rms_hyper_red(k,idx)  = rmse(ker(X_test, x_ctrs) * a_red, f_test);
        cond_hyper_red(k,idx) = cond(sqrt(W_red) * VZ_red);
        leb_hyper_red(k,idx)  = leb_svd(K_leb, VZ_red, W_red, 1e-12);

    end
end

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id = 1;

col_poly = [0.49 0.18 0.56];
col_full = [0.85 0.33 0.10];
col_red  = [0.93 0.69 0.13];
col_ref1 = [0.00 0.00 0.00];

    function pub_axes(ax)
        set(ax, 'FontSize',13,'FontWeight','bold','LineWidth',1.5, ...
            'TickLength',[0.015 0.025],'TickDir','out','Box','on', ...
            'Color','w','XColor','k','YColor','k', ...
            'GridAlpha',0.25,'MinorGridAlpha',0.12);
        set(get(ax,'Parent'),'Color','w');
    end
    function pub_legend(lg)
        set(lg,'Color','w','EdgeColor','k','TextColor','k', ...
            'FontSize',9.5,'FontWeight','normal','LineWidth',1,'Interpreter','tex');
    end

% =========================================================================
%  BLOCK I — CONDITION NUMBER
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_poly_vec, cond_poly,            '-^',  'Color',col_poly,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_poly);
    plot(n_ctrsVec,  cond_hyper_full(k,:), '--s', 'Color',col_full,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec,  cond_hyper_red(k,:),  '-.d', 'Color',col_red, 'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);
    set(gca, 'YScale', 'log'); pub_axes(gca);
    xlabel('n (kernel) / n_{poly}=q+1 (polynomial)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Condition Number (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Poly Hyper','Hyper Full','Hyper Reduced','Location','northwest');
    pub_legend(lg);
    title(sprintf('f_4 - Condition Number - %s - C=%.1f*h_X - %s points', ...
        rbf_names(k), C_ep_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK II — RMSE  (log-log)
% =========================================================================
for k = 1:nk
    tau = tau_vec(k);

    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_poly_vec, rms_poly,            '-^',  'Color',col_poly,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_poly);
    plot(n_ctrsVec,  rms_hyper_full(k,:), '--s', 'Color',col_full,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec,  rms_hyper_red(k,:),  '-.d', 'Color',col_red, 'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    log_n_asym       = log(n_ctrsVec(anchor_idx:end)');
    log_np_asym      = log(n_poly_vec(anchor_idx:end)');
    slope_poly       = fit_slope(log_np_asym, log(rms_poly(anchor_idx:end)'));
    slope_hf         = fit_slope(log_n_asym, log(rms_hyper_full(k,anchor_idx:end)'));
    slope_hr         = fit_slope(log_n_asym, log(rms_hyper_red(k,anchor_idx:end)'));
    slope_hyp_theory = -(tau - 0.5);

    n_anchor  = n_ctrsVec(anchor_idx);
    n_ref_vec = logspace(log10(n_anchor), log10(n_ctrsVec(end)), 60);
    ref_hyp   = rms_hyper_full(k,anchor_idx) * (n_ref_vec / n_anchor).^slope_hyp_theory;
    plot(n_ref_vec, ref_hyp, '-', 'Color',col_ref1, 'LineWidth',1.2);

    set(gca, 'XScale','log', 'YScale','log'); pub_axes(gca);
    xlabel('n (kernel) / n_{poly}=q+1 (polynomial)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Poly Hyper O(n^{%.2f})', slope_poly);
    leg2 = sprintf('Hyper Full     O(n^{%.2f})', slope_hf);
    leg3 = sprintf('Hyper Reduced  O(n^{%.2f})', slope_hr);
    leg4 = sprintf('Ref  O(n^{%.2f})',           slope_hyp_theory);
    lg   = legend(leg1,leg2,leg3,leg4,'Location','southwest');
    pub_legend(lg);
    title(sprintf('f_4 - RMSE - %s  \\tau=%d - ep=%.1f*h_X - %s points', ...
        rbf_names(k), tau, C_ep_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[RMSE slopes]  %s | C_ep=%.2f\n', rbf_names(k), C_ep_vec(k));
    fprintf('  %-48s  %+.4f  (theory: %+.4f)\n', 'Hyper Full:',    slope_hf, slope_hyp_theory);
    fprintf('  %-48s  %+.4f  (theory: %+.4f)\n', 'Hyper Reduced:', slope_hr, slope_hyp_theory);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK III — LEBESGUE CONSTANT  (log-log, SVD-consistent)
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_poly_vec, leb_poly,            '-^',  'Color',col_poly,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_poly);
    plot(n_ctrsVec,  leb_hyper_full(k,:), '--s', 'Color',col_full,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec,  leb_hyper_red(k,:),  '-.d', 'Color',col_red, 'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    all_n   = sort([n_poly_vec, n_ctrsVec]);
    ref_leb = leb_poly(1) * (all_n / all_n(1)).^0.5;
    plot(all_n, ref_leb, '-', 'Color',col_ref1, 'LineWidth',1.2);

    log_n_asym  = log(n_ctrsVec(anchor_idx:end)');
    log_np_asym = log(n_poly_vec(anchor_idx:end)');
    slope_lpoly = fit_slope(log_np_asym, log(leb_poly(anchor_idx:end)'));
    slope_lhf   = fit_slope(log_n_asym, log(leb_hyper_full(k,anchor_idx:end)'));
    slope_lhr   = fit_slope(log_n_asym, log(leb_hyper_red(k,anchor_idx:end)'));

    set(gca, 'XScale','log', 'YScale','log'); pub_axes(gca);
    xlabel('n (kernel) / n_{poly}=q+1 (polynomial)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Poly Hyper     O(n^{%+.2f})', slope_lpoly);
    leg2 = sprintf('Hyper Full     O(n^{%+.2f})', slope_lhf);
    leg3 = sprintf('Hyper Reduced  O(n^{%+.2f})', slope_lhr);
    leg4 = 'Ref  O(n^{+0.50})';
    lg   = legend(leg1,leg2,leg3,leg4,'Location','northwest');
    pub_legend(lg);
    title(sprintf('f_4 - Lebesgue Constant - %s - ep=%.1f*h_X - %s points', ...
        rbf_names(k), C_ep_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[Lebesgue slopes]  %s | C_ep=%.2f\n', rbf_names(k), C_ep_vec(k));
    fprintf('  %-48s  %+.4f\n', 'Hyper Full:',    slope_lhf);
    fprintf('  %-48s  %+.4f\n', 'Hyper Reduced:', slope_lhr);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);

%% =========================================================================
%  DIAGNOSTIC — adaptive ep table
% =========================================================================
fprintf('\n--- Adaptive ep table (d=1, Uniform) ---\n');
fprintf('%-6s', 'n');
for k = 1:nk, fprintf('  %-16s', sprintf('ep(%s)', rbf_names(k))); end
fprintf('\n');
for idx = 1:nn
    fprintf('%-6d', n_ctrsVec(idx));
    for k = 1:nk, fprintf('  %-16.3f', ep_used(k,idx)); end
    fprintf('\n');
end
