clear; clc; close all

%% =========================================================================
%  TEST 2 — FRANKE FUNCTION  (dim = 2)   [corrected]

% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 2;
data_distribution = "Halton";
solver            = "SVD";

% Kernels and CORRECT Sobolev smoothness orders for d=2:  tau = nu + d/2
%   LMatern (nu=3/2): tau = 3/2 + 1 = 5/2
%   QMatern (nu=5/2): tau = 5/2 + 1 = 7/2
rbf_names = ["LMatern", "QMatern"];
tau_vec   = [5/2, 7/2];

C_ep   = 100;
ep_min = 3;
ep_max = 30;

n_ctrsVec = [15, 22, 30, 45, 60, 80, 105, 140];

%% ------------------------------------------------------------------------
%  Test function: Franke function on [0,1]^2
% ------------------------------------------------------------------------
f = @(x) franke(x(:,1), x(:,2));

n_test = 300;
X_test = get_ctrs(n_test, "Uniform", dim);
f_test = f(X_test);

n_leb_1d = 50;
t_leb    = linspace(0, 1, n_leb_1d);
[G1, G2] = meshgrid(t_leb, t_leb);
X_leb    = [G1(:), G2(:)];
n_leb    = size(X_leb, 1);

%% ------------------------------------------------------------------------
%  DIAGNOSTIC: fill distance table for d=2
% ------------------------------------------------------------------------
fprintf('--- Fill distance table (d=2) ---\n');
fprintf('%-6s  %-10s  %-10s\n', 'n', 'h_X', 'ep');
for n = n_ctrsVec
    x   = haltonseq(n, dim);
    h_x = fill_distance(x, 100);
    ep  = C_ep * h_x;
    fprintf('%-6d  %-10.4f  %-10.3f\n', n, h_x, ep);
end
fprintf('\n');

%% -------------------------------------------------------------------------
%  STORAGE  —  2D: (kernel x n)
% -------------------------------------------------------------------------
nk = length(rbf_names);
nn = length(n_ctrsVec);

rms_interp     = zeros(nk, nn);
rms_uls        = zeros(nk, nn);
rms_hyper_full = zeros(nk, nn);
rms_hyper_red  = zeros(nk, nn);

cond_interp     = zeros(nk, nn);
cond_uls        = zeros(nk, nn);
cond_hyper_full = zeros(nk, nn);
cond_hyper_red  = zeros(nk, nn);

leb_interp     = zeros(nk, nn);
leb_uls        = zeros(nk, nn);
leb_hyper_full = zeros(nk, nn);
leb_hyper_red  = zeros(nk, nn);

M_full_vec   = zeros(nk, nn);
ep_used      = zeros(nk, nn);

% Honest exactness diagnostics for the "full" scheme
dimVV_vec    = zeros(nk, nn);   % target dim of V_n x V_n = 1 + n + n(n+1)/2
rank_full    = zeros(nk, nn);   % achieved numerical rank of the product matrix
exact_full   = false(nk, nn);   % whether achieved rank reached the target dim

%% -------------------------------------------------------------------------
%  MAIN LOOP
% -------------------------------------------------------------------------
for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);

    fprintf('\n=== Kernel: %s | C_ep=%.2f | tau=%.2f ===\n', rbf_name, C_ep, tau_vec(k));

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);

        h_X            = fill_distance(x_ctrs, 100);
        ep             = min(max(C_ep * h_X, ep_min), ep_max);
        ep_used(k,idx) = ep;

        ker = @(x, y) rbf(pdist2(x, y), ep);

        fprintf('  n = %3d  |  h_X = %.4f  |  ep = %6.2f\n', n_ctrs, h_X, ep);

        % Target product-space dimension (what full exactness WOULD require)
        dimVV = 1 + n_ctrs + n_ctrs*(n_ctrs+1)/2;
        dimVV_vec(k,idx) = dimVV;

        % ------------------------------------------------------------------
        %  Shared setup: cubature points Z_full
        % ------------------------------------------------------------------
        reduced_param = 0;
        y        = moment(ker, x_ctrs, reduced_param);
        r_target = size(y, 1);
        N_init   = 10 * (n_ctrs + 1);
        meas     = 1;

        [Phi_full, Z_full, w_full, ~, r_ach] = getCF(r_target, N_init, x_ctrs, ...
            ker, meas, y, reduced_param, dim);

        rank_full(k,idx)  = r_ach;
        exact_full(k,idx) = (r_ach >= dimVV);

        M_full             = length(w_full);
        M_full_vec(k, idx) = M_full;

        VZ_full  = ker(Z_full, x_ctrs);
        W_full   = diag(w_full);
        rhs_full = f(Z_full);

        K_leb = ker(X_leb, x_ctrs);

        % ------------------------------------------------------------------
        %  (A) INTERPOLATION AT X
        % ------------------------------------------------------------------
        K_XX = ker(x_ctrs, x_ctrs);
        f_X  = f(x_ctrs);

        [L_X, flag_X] = chol(K_XX, 'lower');
        if flag_X ~= 0
            warning('K(X,X) not SPD at n=%d, ep=%.2f — using backslash', n_ctrs, ep);
            a_interp  = K_XX \ f_X;
            Phi_leb_X = (K_XX \ K_leb')';
        else
            a_interp  = L_X' \ (L_X \ f_X);
            Phi_leb_X = (L_X' \ (L_X \ K_leb'))';
        end

        f_pred_interp        = ker(X_test, x_ctrs) * a_interp;
        rms_interp(k,idx)    = rmse(f_pred_interp, f_test);
        cond_interp(k,idx)   = cond(K_XX);
        leb_interp(k,idx)    = max(sum(abs(Phi_leb_X), 2));

        % ------------------------------------------------------------------
        %  (B) LEAST SQUARES  (unweighted, W = I)
        % ------------------------------------------------------------------
        W_uls = eye(M_full);
        a_uls = LS_stable_solver(W_uls, VZ_full, rhs_full, solver);

        f_pred_uls      = ker(X_test, x_ctrs) * a_uls;
        rms_uls(k,idx)  = rmse(f_pred_uls, f_test);
        cond_uls(k,idx) = cond(VZ_full);

        % SVD-consistent Lebesgue (W = I)
        leb_uls(k,idx) = leb_svd(K_leb, VZ_full, W_uls, 1e-12);

        % ------------------------------------------------------------------
        %  (C) HYPERINTERPOLATION — FULL EXACTNESS (numerical-rank subspace)
        % ------------------------------------------------------------------
        a_full = LS_stable_solver(W_full, VZ_full, rhs_full, solver);

        f_pred_hf             = ker(X_test, x_ctrs) * a_full;
        rms_hyper_full(k,idx) = rmse(f_pred_hf, f_test);
        cond_hyper_full(k,idx)= cond(sqrt(W_full) * VZ_full);

        % SVD-consistent Lebesgue (replaces VtWV backslash)
        leb_hyper_full(k,idx) = leb_svd(K_leb, VZ_full, W_full, 1e-12);

        % ------------------------------------------------------------------
        %  (D) HYPERINTERPOLATION — REDUCED EXACTNESS
        % ------------------------------------------------------------------
        reduced_param = 1;
        y_red    = moment(ker, x_ctrs, reduced_param);
        r_red    = size(y_red, 1);
        N_init_r = 2 * (n_ctrs + 1);

        [~, Z_red, w_red, ~, ~] = getCF(r_red, N_init_r, x_ctrs, ...
            ker, meas, y_red, reduced_param, dim);

        VZ_red  = ker(Z_red, x_ctrs);
        W_red   = diag(w_red);
        rhs_red = f(Z_red);

        a_red = LS_stable_solver(W_red, VZ_red, rhs_red, solver);

        f_pred_hr             = ker(X_test, x_ctrs) * a_red;
        rms_hyper_red(k,idx)  = rmse(f_pred_hr, f_test);
        cond_hyper_red(k,idx) = cond(sqrt(W_red) * VZ_red);

        leb_hyper_red(k,idx)  = leb_svd(K_leb, VZ_red, W_red, 1e-12);

        fprintf(['    M_full = %d  |  M_red = %d  |  dim(VxV)=%d  rank=%d  ' ...
                 'full-exact: %d\n'], ...
                 M_full, length(w_red), dimVV, r_ach, exact_full(k,idx));

    end % n loop
end % kernel loop

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id     = 1;
anchor_idx = 2;

col_interp = [0.47 0.67 0.19];
col_uls    = [0.00 0.45 0.74];
col_full   = [0.85 0.33 0.10];
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
    plot(n_ctrsVec, cond_interp(k,:),     '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, cond_uls(k,:),        '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, cond_hyper_full(k,:), '--s', 'Color',col_full,  'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec, cond_hyper_red(k,:),  '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);
    set(gca,'YScale','log'); pub_axes(gca);
    xlabel('n','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Condition Number (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Interpolation','Least Squares','Hyper Full','Hyper Reduced','Location','northwest');
    pub_legend(lg);
    title(sprintf('f_2 - Condition Number - %s - ep=%.2f*h_X - %s points', ...
        rbf_names(k), C_ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK II — RMSE log-log + empirical slopes  (CORRECTED tau)
% =========================================================================
for k = 1:nk
    tau          = tau_vec(k);
    slope_theory = -(tau - 1) / 2;   % Eq.(19), d=2: -(tau-d/2)/d  with d=2

    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    rmse_i  = rms_interp(k,:);
    rmse_u  = rms_uls(k,:);
    rmse_hf = rms_hyper_full(k,:);
    rmse_hr = rms_hyper_red(k,:);

    plot(n_ctrsVec, rmse_i,  '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, rmse_u,  '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, rmse_hf, '--s', 'Color',col_full,  'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec, rmse_hr, '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_i    = fit_slope(log_n_asym, log(rmse_i(anchor_idx:end)'));
    slope_u    = fit_slope(log_n_asym, log(rmse_u(anchor_idx:end)'));
    slope_hf   = fit_slope(log_n_asym, log(rmse_hf(anchor_idx:end)'));
    slope_hr   = fit_slope(log_n_asym, log(rmse_hr(anchor_idx:end)'));

    n_anchor  = n_ctrsVec(anchor_idx);
    n_ref_vec = logspace(log10(n_anchor), log10(n_ctrsVec(end)), 60);
    ref_line  = rmse_hf(anchor_idx) * (n_ref_vec / n_anchor).^slope_theory;
    plot(n_ref_vec, ref_line, '-', 'Color',col_ref1, 'LineWidth',1.2);

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Interpolation  O(n^{%.2f})', slope_i);
    leg2 = sprintf('Least Squares  O(n^{%.2f})', slope_u);
    leg3 = sprintf('Hyper Full     O(n^{%.2f})', slope_hf);
    leg4 = sprintf('Hyper Reduced  O(n^{%.2f})', slope_hr);
    leg5 = sprintf('Ref  O(n^{%.2f})',           slope_theory);
    lg   = legend(leg1,leg2,leg3,leg4,leg5,'Location','northeast');
    pub_legend(lg);
    title(sprintf('f_2 - RMSE - %s  \\tau=%.1f - d=2 - ep=%.1f*h_X - %s points', ...
        rbf_names(k), tau, C_ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[RMSE slopes]  %s | tau=%.2f | C_ep=%.2f\n', rbf_names(k), tau, C_ep);
    fprintf('  %-48s  %+.4f\n',                  'Interpolation:', slope_i);
    fprintf('  %-48s  %+.4f\n',                  'Least Squares:', slope_u);
    fprintf('  %-48s  %+.4f  (theory: %+.4f)\n', 'Hyper Full:',    slope_hf, slope_theory);
    fprintf('  %-48s  %+.4f  (theory: %+.4f)\n', 'Hyper Reduced:', slope_hr, slope_theory);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK III — LEBESGUE CONSTANT log-log  (SVD-consistent values)
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    leb_i  = leb_interp(k,:);
    leb_u  = leb_uls(k,:);
    leb_hf = leb_hyper_full(k,:);
    leb_hr = leb_hyper_red(k,:);

    plot(n_ctrsVec, leb_i,  '-^',  'Color',col_interp,'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, leb_u,  '-o',  'Color',col_uls,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, leb_hf, '--s', 'Color',col_full,  'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_full);
    plot(n_ctrsVec, leb_hr, '-.d', 'Color',col_red,   'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    ref_leb = leb_i(1) * (n_ctrsVec / n_ctrsVec(1)).^0.5;
    plot(n_ctrsVec, ref_leb, '-', 'Color',col_ref1, 'LineWidth',1.2);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_li   = fit_slope(log_n_asym, log(leb_i(anchor_idx:end)'));
    slope_lu   = fit_slope(log_n_asym, log(leb_u(anchor_idx:end)'));
    slope_lhf  = fit_slope(log_n_asym, log(leb_hf(anchor_idx:end)'));
    slope_lhr  = fit_slope(log_n_asym, log(leb_hr(anchor_idx:end)'));

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Interpolation','Least Squares','Hyper Full','Hyper Reduced','Ref  O(n^{+0.50})','Location','northwest');
    pub_legend(lg);
    title(sprintf('f_2 - Lebesgue Constant - %s - ep=%.1f*h_X  d=2 - %s points', ...
        rbf_names(k), C_ep, data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[Lebesgue slopes]  %s | C_ep=%.2f\n', rbf_names(k), C_ep);
    fprintf('  %-48s  %+.4f\n', 'Interpolation:', slope_li);
    fprintf('  %-48s  %+.4f\n', 'Least Squares:', slope_lu);
    fprintf('  %-48s  %+.4f\n', 'Hyper Full:',    slope_lhf);
    fprintf('  %-48s  %+.4f\n', 'Hyper Reduced:', slope_lhr);

    savefig_png(fig_id); fig_id = fig_id + 1;
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);

%% =========================================================================
%  DIAGNOSTICS  (incl. honest full-exactness reporting)
% =========================================================================
fprintf('\n--- Full-exactness reality check ---\n');
fprintf('%-6s  %-12s  %-10s  %-10s\n', 'n', 'dim(VxV)', 'rank', 'exact?');
for k = 1:nk
    fprintf('Kernel %s:\n', rbf_names(k));
    for idx = 1:nn
        fprintf('%-6d  %-12d  %-10d  %-10d\n', ...
            n_ctrsVec(idx), dimVV_vec(k,idx), rank_full(k,idx), exact_full(k,idx));
    end
end

fprintf('\n--- Adaptive ep table (d=2) ---\n');
fprintf('%-6s', 'n');
for k = 1:nk, fprintf('  %-16s', sprintf('ep(%s)', rbf_names(k))); end
fprintf('\n');
for idx = 1:nn
    fprintf('%-6d', n_ctrsVec(idx));
    for k = 1:nk, fprintf('  %-16.3f', ep_used(k,idx)); end
    fprintf('\n');
end

fprintf('\n--- Cubature size M_full vs n ---\n');
fprintf('%-6s', 'n');
for k = 1:nk, fprintf('  %-16s', sprintf('M_full(%s)', rbf_names(k))); end
fprintf('\n');
for idx = 1:nn
    fprintf('%-6d', n_ctrsVec(idx));
    for k = 1:nk, fprintf('  %-16d', M_full_vec(k,idx)); end
    fprintf('\n');
end
