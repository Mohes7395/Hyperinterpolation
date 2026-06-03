clear; clc; close all

%% =========================================================================
%  TEST 5 — LARGE-n REGIME, REDUCED EXACTNESS WITH CLOSED-FORM MOMENTS
%
%  Purpose. Tests 1--4 are capped at moderate n because FULL exactness drags
%  in the O(n^2) product space. REDUCED exactness (CFv) imposes exactness only
%  on (N_K(X)+1) x 1, whose RHS is the vector of SINGLE-kernel moments
%  b_i = \int_0^1 K(x,x_i) dx. On [0,1] these moments are available in closed
%  form (moment_closed_1d.m), so the construction is cheap and exact and n can
%  be pushed well beyond the earlier range. This isolates the behavior of the
%  METHODS at large n, free of any moment-estimation error.
%
%  Methods compared (all living in N_K(X)):
%    (A) Kernel interpolation at X.
%    (B) Kernel least squares (centers X, sampled at Z, unweighted).
%    (C) Reduced kernel hyperinterpolation (CFv), weights from the
%        Backus--Gilbert solve with the CLOSED-FORM moment RHS.
%
%  Primary quantity: Lebesgue constant vs n (slope over a wide n-window).
%  Secondary quantity: RMSE vs n.
%
%  Kernels: Linear and Quadratic Matern (d=1).  Domain: [0,1].
%
%  NOTE on conditioning. Pushing n large does NOT remove kernel-matrix
%  ill-conditioning: with ep = C*h_X and h_X -> 0 the systems become severely
%  ill-conditioned, and the SVD truncation (1e-12) eventually sets an error
%  floor. The slope fits use an asymptotic window (anchor_idx) chosen before
%  that floor dominates.
% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 1;
data_distribution = "Halton";
solver            = "SVD";

rbf_names = ["LMatern", "QMatern"];
tau_vec   = [2, 3];           % Sobolev smoothness orders (d=1)

% Large-n grid (log-spaced); well beyond the n<=200 of Tests 1-4.
n_ctrsVec = [50, 80, 120, 180, 260, 360, 480, 600, 750, 900, 1080, 1620];
anchor_idx = 2;               % skip the first point in slope fits

% Adaptive shape parameter ep = C_ep * h_X, with a floor/ceiling.
C_ep   = 500;
ep_min = 3;
ep_max = 50;

%% ------------------------------------------------------------------------
%  Test function and evaluation grids
% ------------------------------------------------------------------------
f = @(x) sin(2*pi*x) + 0.5*sin(4*pi*x);

n_test = 500;
X_test = get_ctrs(n_test, "Uniform", dim);
f_test = f(X_test);

n_leb = 1000;
X_leb = linspace(0, 1, n_leb)';

%% -------------------------------------------------------------------------
%  STORAGE  —  (kernel x n)
% -------------------------------------------------------------------------
nk = length(rbf_names);
nn = length(n_ctrsVec);

rms_interp    = zeros(nk, nn);
rms_uls       = zeros(nk, nn);
rms_hyper_red = zeros(nk, nn);

cond_interp    = zeros(nk, nn);
cond_uls       = zeros(nk, nn);
cond_hyper_red = zeros(nk, nn);

leb_interp     = zeros(nk, nn);
leb_uls        = zeros(nk, nn);
leb_hyper_red  = zeros(nk, nn);

M_red_vec = zeros(nk, nn);
ep_used   = zeros(nk, nn);

%% -------------------------------------------------------------------------
%  MAIN LOOP
% -------------------------------------------------------------------------
for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);

    fprintf('\n=== Kernel: %s ===\n', rbf_name);

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);

        h_X            = fill_distance(x_ctrs, 5000);
        % ep           = min(max(ep_min, C_ep * h_X), ep_max);
        ep             = 3;
        ep_used(k,idx) = ep;

        ker = @(x, y) rbf(pdist2(x, y), ep);

        fprintf('  n = %4d  |  h_X = %.5f  |  ep = %7.2f\n', n_ctrs, h_X, ep);

        K_leb = ker(X_leb, x_ctrs);   % n_leb x n

        % ------------------------------------------------------------------
        %  (A) INTERPOLATION AT X
        % ------------------------------------------------------------------
        K_XX = ker(x_ctrs, x_ctrs);
        f_X  = f(x_ctrs);

        [L_X, flag_X] = chol(K_XX, 'lower');
        if flag_X ~= 0
            a_interp  = K_XX \ f_X;
            Phi_leb_X = (K_XX \ K_leb')';
        else
            a_interp  = L_X' \ (L_X \ f_X);
            Phi_leb_X = (L_X' \ (L_X \ K_leb'))';
        end

        rms_interp(k,idx)  = rmse(ker(X_test, x_ctrs) * a_interp, f_test);
        cond_interp(k,idx) = cond(K_XX);
        leb_interp(k,idx)  = max(sum(abs(Phi_leb_X), 2));

        % ------------------------------------------------------------------
        %  (C-prep) REDUCED CUBATURE with CLOSED-FORM moment RHS
        %
        %  RHS b is computed analytically (moment_closed_1d), ordered
        %  [const ; kernels] to match prod_basis(...,reduced_param=1)'.
        %  getCF then performs the Backus--Gilbert / NNLS solve for W.
        % ------------------------------------------------------------------
        reduced_param = 1;
        b_red    = moment_closed_1d(rbf_name, x_ctrs, ep);   % (n+1) x 1, exact
        r_red    = numel(b_red);
        N_init_r = 2 * (n_ctrs + 1);
        meas     = 1;

        [~, Z_red, w_red, ~, ~] = getCF(r_red, N_init_r, x_ctrs, ...
            ker, meas, b_red, reduced_param, dim);

        M_red_vec(k,idx) = length(w_red);

        VZ_red  = ker(Z_red, x_ctrs);
        W_red   = diag(w_red);

        % ------------------------------------------------------------------
        %  (B) LEAST SQUARES  (unweighted, same nodes Z_red)
        % ------------------------------------------------------------------
        M_red  = length(w_red);
        W_uls  = eye(M_red);
        a_uls  = LS_stable_solver(W_uls, VZ_red, f(Z_red), solver);

        rms_uls(k,idx)  = rmse(ker(X_test, x_ctrs) * a_uls, f_test);
        cond_uls(k,idx) = cond(VZ_red);
        leb_uls(k,idx)  = leb_svd(K_leb, VZ_red, W_uls, 1e-12);

        % ------------------------------------------------------------------
        %  (C) REDUCED HYPERINTERPOLATION
        % ------------------------------------------------------------------
        a_red = LS_stable_solver(W_red, VZ_red, f(Z_red), solver);

        rms_hyper_red(k,idx)  = rmse(ker(X_test, x_ctrs) * a_red, f_test);
        cond_hyper_red(k,idx) = cond(sqrt(W_red) * VZ_red);
        leb_hyper_red(k,idx)  = leb_svd(K_leb, VZ_red, W_red, 1e-12);

        fprintf('    M_red = %d\n', M_red);

    end
end

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id = 1;

col_interp = [0.47 0.67 0.19];
col_uls    = [0.00 0.45 0.74];
col_red    = [0.93 0.69 0.13];
col_ref1   = [0.00 0.00 0.00];

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
%  BLOCK I — CONDITION NUMBER vs n
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_ctrsVec, cond_interp(k,:),    '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, cond_uls(k,:),       '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, cond_hyper_red(k,:), '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);
    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Condition Number (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Interpolation','Least Squares','Hyper Reduced','Location','northwest');
    pub_legend(lg);
    title(sprintf('f_1 - Condition Number - %s - ep=%.1f - %s points (large n)', ...
        rbf_names(k), ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK II — RMSE vs n  (secondary)
% =========================================================================
for k = 1:nk
    tau = tau_vec(k);
    slope_theory = -(tau - 0.5);   % Eq.(20), d=1

    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    rmse_i  = rms_interp(k,:);
    rmse_u  = rms_uls(k,:);
    rmse_hr = rms_hyper_red(k,:);

    plot(n_ctrsVec, rmse_i,  '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, rmse_u,  '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, rmse_hr, '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_i    = fit_slope(log_n_asym, log(rmse_i(anchor_idx:end)'));
    slope_u    = fit_slope(log_n_asym, log(rmse_u(anchor_idx:end)'));
    slope_hr   = fit_slope(log_n_asym, log(rmse_hr(anchor_idx:end)'));

    n_anchor  = n_ctrsVec(anchor_idx);
    n_ref_vec = logspace(log10(n_anchor), log10(n_ctrsVec(end)), 60);
    ref_line  = rmse_hr(anchor_idx) * (n_ref_vec / n_anchor).^slope_theory;
    plot(n_ref_vec, ref_line, '-', 'Color',col_ref1, 'LineWidth',1.2);

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Interpolation  O(n^{%.2f})', slope_i);
    leg2 = sprintf('Least Squares  O(n^{%.2f})', slope_u);
    leg3 = sprintf('Hyper Reduced  O(n^{%.2f})', slope_hr);
    leg4 = sprintf('Ref  O(n^{%.2f})',           slope_theory);
    lg   = legend(leg1,leg2,leg3,leg4,'Location','southwest');
    pub_legend(lg);
    title(sprintf('f_1 - RMSE - %s  \\tau=%d - ep=%.1f - %s points (large n)', ...
        rbf_names(k), tau, ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[RMSE slopes]  %s\n', rbf_names(k));
    fprintf('  %-20s %+.4f\n', 'Interpolation:', slope_i);
    fprintf('  %-20s %+.4f\n', 'Least Squares:', slope_u);
    fprintf('  %-20s %+.4f\n', 'Hyper Reduced:', slope_hr);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK III — LEBESGUE CONSTANT vs n  (PRIMARY)
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    leb_i  = leb_interp(k,:);
    leb_u  = leb_uls(k,:);
    leb_hr = leb_hyper_red(k,:);

    plot(n_ctrsVec, leb_i,  '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, leb_u,  '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, leb_hr, '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    ref_leb = leb_i(1) * (n_ctrsVec / n_ctrsVec(1)).^0.5;
    plot(n_ctrsVec, ref_leb, '-', 'Color',col_ref1, 'LineWidth',1.2);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_li   = fit_slope(log_n_asym, log(leb_i(anchor_idx:end)'));
    slope_lu   = fit_slope(log_n_asym, log(leb_u(anchor_idx:end)'));
    slope_lhr  = fit_slope(log_n_asym, log(leb_hr(anchor_idx:end)'));

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Interpolation  O(n^{%+.2f})', slope_li);
    leg2 = sprintf('Least Squares  O(n^{%+.2f})', slope_lu);
    leg3 = sprintf('Hyper Reduced  O(n^{%+.2f})', slope_lhr);
    leg4 = 'Ref  O(n^{+0.50})';
    lg   = legend(leg1,leg2,leg3,leg4,'Location','northwest');
    pub_legend(lg);
    title(sprintf('f_1 - Lebesgue Constant - %s - ep=%.1f - %s points (large n)', ...
        rbf_names(k), ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[Lebesgue slopes]  %s\n', rbf_names(k));
    fprintf('  %-20s %+.4f\n', 'Interpolation:', slope_li);
    fprintf('  %-20s %+.4f\n', 'Least Squares:', slope_lu);
    fprintf('  %-20s %+.4f\n', 'Hyper Reduced:', slope_lhr);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);

%% =========================================================================
%  DIAGNOSTICS
% =========================================================================
fprintf('\n--- ep and M_red vs n ---\n');
fprintf('%-6s', 'n');
for k = 1:nk, fprintf('  %-14s %-10s', sprintf('ep(%s)',rbf_names(k)), sprintf('M(%s)',rbf_names(k))); end
fprintf('\n');
for idx = 1:nn
    fprintf('%-6d', n_ctrsVec(idx));
    for k = 1:nk, fprintf('  %-14.3f %-10d', ep_used(k,idx), M_red_vec(k,idx)); end
    fprintf('\n');
end
