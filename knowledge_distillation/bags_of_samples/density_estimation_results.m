% Shows the results of the density estimation demo.
%
% George Papamakarios, Aug 2015

clear;
close all;

for i = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

    % load results
    load(fullfile('outdir', 'bags_of_samples', sprintf('density_demo_results_%d.mat', i)));

    % parameters for the plots
    fontsize = 18;
    linewidth = 3;
    savedir = fullfile('..', 'reports', 'figs', 'compact_predictive');
    w_range = [-7, 7];
    x_range = [-15, 10];
    col = distinguishable_colors(5);

    % plot mcmc posterior marginals of the weights
    figure;
    for k = 1:K
        subplot(1, K, k); hold on;
        [mm, marg_post_m] = samples2pdf(ms_post(k,:));
        plot(m_true(k)*[1 1], [0 2], '--', 'Color', [139, 149, 173]/255, 'LineWidth', 2);
        plot(mm, marg_post_m, 'r', 'LineWidth', linewidth); 
        xlabel(sprintf('m_%d', k), 'FontSize', fontsize);
        xlim(w_range);
        ylim([0, 2]);
        set(gca, 'YTick', []);
        set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
        set(gca, 'FontSize', fontsize);
    end
    filename = fullfile(savedir, sprintf('density_mcmc_hist_%d', ns_y_true));
    print([filename, '.eps'], '-depsc');
    system(['epstopdf ', filename, '.eps']);
    system(['rm -f ', filename, '.eps']);

    % plot true and predictive pdfs
    figure; hold on;
    yy = linspace(x_range(1), x_range(2));
    pdf_true = exp(log_mog_pdf(yy, a_true, m_true, s_true));
    pdf_pred = exp(log_mog_pdf(yy, a_pred, m_pred, s_pred));
    pdf_comp = exp(log_mog_pdf(yy, a_comp, m_comp, s_comp));
    pdf_onln = exp(log_mog_pdf(yy, a_onln, m_onln, s_onln));
    plot(yy, pdf_true, 'Color', col(1,:), 'LineWidth', linewidth);
    plot(yy, pdf_pred, 'Color', col(2,:), 'LineWidth', linewidth);
    plot(yy, pdf_comp, 'Color', col(3,:), 'LineWidth', linewidth);
    plot(yy, pdf_onln, 'Color', col(4,:), 'LineWidth', linewidth);
    plot(ys_true, 0, 'x', 'Color', col(5,:), 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('x', 'FontSize', fontsize);
    xlim(x_range);
    ylim([0, 0.2]);
    set(gca, 'YTick', []);
    set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
    legend('True model', 'Monte Carlo', 'Compact (batch)', 'Compact (online)', 'Data points', 'Location', 'NorthWest');
    set(gca, 'FontSize', fontsize);
    filename = fullfile(savedir, sprintf('density_predictive_%d', ns_y_true));
    print([filename, '.eps'], '-depsc');
    system(['epstopdf ', filename, '.eps']);
    system(['rm -f ', filename, '.eps']);
end
