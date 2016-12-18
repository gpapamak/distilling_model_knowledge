% Plots and saves the results of the binary classification demo.
%
% George Papamakarios, Aug 2015

clear;
close all;

load(fullfile('outdir', 'bags_of_samples', 'classification_demo_results.mat'));

% options
xmin = -8; xmax = 8;
[X, Y] = meshgrid(linspace(xmin, xmax, 200));
wmin = -23; wmax = 23;
[Wx, Wy] = meshgrid(linspace(wmin, wmax, 200));
facealpha = 0.35;
fontsize = 16;
savedir = fullfile('..', 'reports', 'figs', 'compact_predictive');

%% setup

% prior
figure; hold on;
Z = reshape(log_gauss_pdf([Wx(:), Wy(:)]', zeros(2,1), prior_var_w*eye(2)), size(Wx));
surfc(Wx, Wy, 2*pi*prior_var_w * exp(Z), 'facealpha', facealpha); shading flat; colorbar;
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_prior');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% dataset
figure; hold on;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 20);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 20);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_dataset');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

%% posteriors

% MCMC
figure; hold on;
perc_samples_to_plot = 0.1;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
ws_post_to_plot = data_sample(ws_post, false, floor(perc_samples_to_plot * ns_w_post));
ws_post_to_plot = ws_post_to_plot(:, ~logical(sum(ws_post_to_plot > wmax | ws_post_to_plot < wmin)));
surfc(Wx, Wy, exp(Z), 'facealpha', facealpha); shading flat; colorbar;
plot(ws_post_to_plot(1,:), ws_post_to_plot(2,:), 'k.');
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_mcmc');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% EP
figure; hold on;
Z = reshape(log_gauss_pdf([Wx(:), Wy(:)]', m_ep, S_ep), size(Wx));
surfc(Wx, Wy, sqrt(det(2*pi*S_ep)) * exp(Z), 'facealpha', facealpha); shading flat; colorbar;
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_ep');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, cross entropy, batch
figure; hold on;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
surfc(Wx, Wy, exp(Z), 'facealpha', facealpha); shading flat; colorbar;
plot(net_comp.weights{1}(:,1), net_comp.weights{1}(:,2), 'k.', 'MarkerSize', 15);
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_ce_batch');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, derivative square error, batch
figure; hold on;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
surfc(Wx, Wy, exp(Z), 'facealpha', facealpha); shading flat; colorbar;
plot(net_comp_derv.weights{1}(:,1), net_comp_derv.weights{1}(:,2), 'k.', 'MarkerSize', 15);
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_dse_batch');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, cross entropy, online
figure; hold on;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
surfc(Wx, Wy, exp(Z), 'facealpha', facealpha); shading flat; colorbar;
plot(net_onln.weights{1}(:,1), net_onln.weights{1}(:,2), 'k.', 'MarkerSize', 15);
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_ce_online');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, derivative square error, online
figure; hold on;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
surfc(Wx, Wy, exp(Z), 'facealpha', facealpha); shading flat; colorbar;
plot(net_onln_derv.weights{1}(:,1), net_onln_derv.weights{1}(:,2), 'k.', 'MarkerSize', 15);
xlabel('w_1', 'FontSize', fontsize);
ylabel('w_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_posterior_dse_online');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

%% predictive distributions

% MCMC
figure; hold on;
Z = reshape(net_post.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_mcmc');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% EP
figure; hold on;
m_1d = [X(:), Y(:)] * m_ep;
S_1d = sum([X(:), Y(:)]' .* (S_ep * [X(:), Y(:)]'), 1);
Z = reshape(gauss_expectation(@(z) sigm(z), m_1d, S_1d), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_ep');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, cross entropy, batch
figure; hold on;
Z = reshape(net_comp.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_ce_batch');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, derivative square error, batch
figure; hold on;
Z = reshape(net_comp_derv.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_dse_batch');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, cross entropy, online
figure; hold on;
Z = reshape(net_onln.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_ce_online');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

% compact, derivative square error, online
figure; hold on;
Z = reshape(net_onln_derv.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', facealpha); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 18);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 18);
xlabel('x_1', 'FontSize', fontsize);
ylabel('x_2', 'FontSize', fontsize);
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
set(gca, 'fontsize', fontsize);
filename = fullfile(savedir, 'logreg_predictive_dse_online');
plot2svg([filename, '.svg']);
system(['inkscape -D -z --file=', filename, '.svg --export-pdf=', filename, '.pdf']);
system(['rm -f ', filename, '.svg']);

%% loose ends
system(['rm -f ', fullfile(savedir, '*.png')]);
