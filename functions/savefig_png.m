function savefig_png(fig_id)
    filename = sprintf('Figure_%d.png', fig_id);
    exportgraphics(gcf, filename, 'Resolution', 300, 'BackgroundColor', 'white');
end