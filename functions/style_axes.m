function style_axes(ax)
% Apply consistent axis styling.
    set(gcf, 'Color', 'w');
    ax.Color        = 'w';
    ax.XColor       = 'k';
    ax.YColor       = 'k';
    ax.LineWidth    = 1;
    ax.GridColor    = [0.8 0.8 0.8];
    ax.GridAlpha    = 0.4;
    ax.MinorGridColor = [0.85 0.85 0.85];
    ax.MinorGridAlpha = 0.3;
    ax.XMinorGrid   = 'off';
    ax.YMinorGrid   = 'on';
    grid(ax, 'on');
    box on;
end