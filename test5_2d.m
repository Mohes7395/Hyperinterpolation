clear; clc; close all

%% =========================================================================
%  TEST 5 (2D) — REDUCED EXACTNESS, FIXED SHAPE PARAMETER
%
%  Two-dimensional counterpart of Test 5. Reduced exactness (CFv) imposes
%  exactness only on (N_K(X)+1) x 1, so the construction costs O(n) constraints
%  and the O(n^2) product-space wall of full exactness is absent. This lets n
%  grow further than in the full-exactness tests.
%
%  Design:
%    * Domain [0,1]^2, Franke function f_2 (as in Test 2).
%    * FIXED shape parameter ep (not ep = C*h_X), set per kernel below.
%    * Moments (reduced RHS) by QMC (moment.m), valid in any dimension.
%    * Three methods: interpolation, least squares, reduced hyperinterpolation.
%
%  Reported: condition number, RMSE, Lebesgue constant vs n.
%
%  Kernels: Linear and Quadratic Matern (d=2).
% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 2;
data_distribution = "Halton";
solver            = "SVD";

rbf_names = ["LMatern", "QMatern"];
% Sobolev smoothness orders in d=2:  tau = nu + d/2
%   LMatern (nu=3/2) -> tau = 5/2 ;  QMatern (nu=5/2) -> tau = 7/2
tau_vec   = [5/2, 7/2];

% FIXED shape parameter, one per kernel (not tied to h_X). Moderate for d=2.
ep_fixed_vec = [1, 1];

% n grid (~6^2 up to ~26^2)
n_ctrsVec  = [36, 64, 100, 144, 225, 324, 484, 676, 946];
anchor_idx = 2;

%% ------------------------------------------------------------------------
%  Test function (Franke) and evaluation grids
% ------------------------------------------------------------------------
f = @(x) franke(x(:,1), x(:,2));

n_test = 300;
X_test = get_ctrs(n_test, "Uniform", dim);
f_test = f(X_test);

% Lebesgue grid: 50x50 tensor grid over [0,1]^2
n_leb_1d = 50;
t_leb    = linspace(0, 1, n_leb_1d);
[G1, G2] = meshgrid(t_leb, t_leb);
X_leb    = [G1(:), G2(:)];
n_leb    = size(X_leb, 1);

%% -------------------------------------------------------------------------
%  STORAGE
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

%% -------------------------------------------------------------------------
%  MAIN LOOP
% -------------------------------------------------------------------------
for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);
    ep       = ep_fixed_vec(k);

    fprintf('\n=== Kernel: %s | FIXED ep = %.2f (d=2) ===\n', rbf_name, ep);

    ker = @(x, y) rbf(pdist2(x, y), ep);

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);

        fprintf('  n = %4d\n', n_ctrs);

        K_leb = ker(X_leb, x_ctrs);   % n_leb x n

        % (A) INTERPOLATION
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

        % (C-prep) REDUCED cubature, QMC moments
        reduced_param = 1;
        y_red    = moment(ker, x_ctrs, reduced_param);
        r_red    = size(y_red, 1);
        N_init_r = 2 * (n_ctrs + 1);
        meas     = 1;

        [~, Z_red, w_red, ~, ~] = getCF(r_red, N_init_r, x_ctrs, ...
            ker, meas, y_red, reduced_param, dim);

        M_red_vec(k,idx) = length(w_red);
        VZ_red = ker(Z_red, x_ctrs);
        W_red  = diag(w_red);
        M_red  = length(w_red);

        % (B) LEAST SQUARES (same nodes, unweighted)
        W_uls = eye(M_red);
        a_uls = LS_stable_solver(W_uls, VZ_red, f(Z_red), solver);
        rms_uls(k,idx)  = rmse(ker(X_test, x_ctrs) * a_uls, f_test);
        cond_uls(k,idx) = cond(VZ_red);
        leb_uls(k,idx)  = leb_svd(K_leb, VZ_red, W_uls, 1e-12);

        % (C) REDUCED HYPERINTERPOLATION
        a_red = LS_stable_solver(W_red, VZ_red, f(Z_red), solver);
        rms_hyper_red(k,idx)  = rmse(ker(X_test, x_ctrs) * a_red, f_test);
        cond_hyper_red(k,idx) = cond(sqrt(W_red) * VZ_red);
        leb_hyper_red(k,idx)  = leb_svd(K_leb, VZ_red, W_red, 1e-12);

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

% BLOCK I — CONDITION NUMBER
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
    title(sprintf('f_2 - Condition Number - %s - ep=%.0f (fixed) - d=2 - %s points', ...
        rbf_names(k), ep_fixed_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% BLOCK II — RMSE
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_ctrsVec, rms_interp(k,:),    '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, rms_uls(k,:),       '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, rms_hyper_red(k,:), '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);
    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Interpolation','Least Squares','Hyper Reduced','Location','best');
    pub_legend(lg);
    title(sprintf('f_2 - RMSE - %s - ep=%.0f (fixed) - d=2 - %s points', ...
        rbf_names(k), ep_fixed_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% BLOCK III — LEBESGUE CONSTANT (primary)
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
    set(gca,'XScale','log','YScale','log'); pub_axes(gca);

    ln        = log(n_ctrsVec(anchor_idx:end)');
    slope_li  = fit_slope(ln, log(leb_i(anchor_idx:end)'));
    slope_lu  = fit_slope(ln, log(leb_u(anchor_idx:end)'));
    slope_lhr = fit_slope(ln, log(leb_hr(anchor_idx:end)'));

    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    leg1 = sprintf('Interpolation  O(n^{%+.2f})', slope_li);
    leg2 = sprintf('Least Squares  O(n^{%+.2f})', slope_lu);
    leg3 = sprintf('Hyper Reduced  O(n^{%+.2f})', slope_lhr);
    leg4 = 'Ref  O(n^{+0.50})';
    lg   = legend(leg1,leg2,leg3,leg4,'Location','northwest');
    pub_legend(lg);
    title(sprintf('f_2 - Lebesgue Constant - %s - ep=%.0f (fixed) - d=2 - %s points', ...
        rbf_names(k), ep_fixed_vec(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

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
fprintf('\n--- kappa(K_XX) and M_red vs n ---\n');
fprintf('%-6s', 'n');
for k = 1:nk
    fprintf('  %-14s %-10s', sprintf('kappa(%s)',rbf_names(k)), sprintf('M(%s)',rbf_names(k)));
end
fprintf('\n');
for idx = 1:nn
    fprintf('%-6d', n_ctrsVec(idx));
    for k = 1:nk
        fprintf('  %-14.2e %-10d', cond_interp(k,idx), M_red_vec(k,idx));
    end
    fprintf('\n');
end
