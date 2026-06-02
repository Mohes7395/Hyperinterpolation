function slope = fit_slope(log_n_col, log_err_col)
% Linear regression log_err = slope * log_n + const
% Returns the slope (convergence rate exponent).
    log_err_col = log_err_col(:);
    valid = isfinite(log_err_col) & isfinite(log_n_col);
    if sum(valid) < 2
        slope = NaN; return;
    end
    p     = polyfit(log_n_col(valid), log_err_col(valid), 1);
    slope = p(1);
end