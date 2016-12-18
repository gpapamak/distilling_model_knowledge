% Bayesian logistic regression by fitting a compact representation to the
% bayesian predictive distribution.
% Based on:
% E. Snelson and Z. Ghahramani, "Compact approximations to
% Bayesian predictive distributions," ICML, 2005.
%
% George Papamakarios, Jan 2015

clear;

% select which classification to use
%class_n = 'probit'; class_f = @probit;
class_n = 'logistic'; class_f = @sigm;

% generate data samples from the true model
%ns_xy_true_perclass = 12;
%w_true = [1 0]';
%xs_true = [gauss_sample([-4 0], eye(2), ns_xy_true_perclass), gauss_sample([4 0], eye(2), ns_xy_true_perclass)];
%ys_true = 2 * (rand(1, 2*ns_xy_true_perclass) < class_f(w_true' * xs_true)) - 1;
load(fullfile('outdir', 'bags_of_samples', 'classification_demo_dataset.mat'));
xys_true = xs_true .* (ones(2, 1) * ys_true);

%% -- fit models

% generate weight samples from the posterior on the weights
prior_var_w = 100;
ns_w_post = 1e+4;
mcmc_burnin = floor(0.1 * ns_w_post);
log_post_w = @(w) sum(log(class_f(w' * xys_true)), 2) - sum(w' .^ 2, 2) / (2 * prior_var_w);
ws_post = slice_sample(log_post_w, zeros(2,1), ns_w_post, 'burnin', mcmc_burnin);
net_post = NeuralNet(2);
net_post.addLayer(ns_w_post, 'type', class_n, 'weights', ws_post', 'biases', zeros(ns_w_post, 1));
net_post.addLayer(1, 'type', 'linear', 'weights', ones(1, ns_w_post) / ns_w_post, 'biases', 0);

% approximate posterior on the weights with EP
[m_ep, S_ep] = ep_gauss(xs_true, ys_true, prior_var_w, 'method', class_n, 'maxepochs', 1e+3, 'verbose', true);

% fit a compact representation
ncomp = 10;
batch_size = 10;
online_reps = 5000;
step = LinearDecay(1, online_reps);
input_gen = @(N) gauss_sample([0 0], 10 * eye(2), N);
net_comp = NeuralNet(2);
net_comp.addLayer(ncomp, 'type', class_n, 'biases', zeros(ncomp, 1), 'fix_biases', true);
net_comp.addLayer(1, 'type', 'linear', 'weights', ones(1, ncomp) / ncomp, 'fix_weights', true, 'biases', 0, 'fix_biases', true);
net_comp.setLossFunction('cross_entropy');
net_comp.train_net(net_post, input_gen, 'step', step, 'minibatch', batch_size, 'maxiter', online_reps);

% fit the compact representation now using derivatives
step = LinearDecay(1, online_reps);
net_comp_derv = NeuralNet(2);
net_comp_derv.addLayer(ncomp, 'type', class_n, 'biases', zeros(ncomp, 1), 'fix_biases', true);
net_comp_derv.addLayer(1, 'type', 'linear', 'weights', ones(1, ncomp) / ncomp, 'fix_weights', true, 'biases', 0, 'fix_biases', true);
net_comp_derv.setLossFunction('avg_score_matching');
net_comp_derv.train_net(net_post, input_gen, 'step', step, 'minibatch', batch_size, 'maxiter', online_reps);

% fit a compact representation online
batch_size = 10;
online_reps = 5000;
mcmc_thin = 1;
stream = BinaryClassificationSampler(xs_true, ys_true, class_f, prior_var_w, zeros(2,1), 10*eye(2), mcmc_burnin, mcmc_thin);
step = LinearDecay(1, online_reps);
net_onln = NeuralNet(2);
net_onln.addLayer(ncomp, 'type', class_n, 'biases', zeros(ncomp, 1), 'fix_biases', true);
net_onln.addLayer(1, 'type', 'linear', 'weights', ones(1, ncomp) / ncomp, 'fix_weights', true, 'biases', 0, 'fix_biases', true);
net_onln.setLossFunction('cross_entropy');
net_onln.train_stream(stream, 'step', step, 'minibatch', batch_size, 'maxiter', online_reps);

% fit a compact representation online using derivatives
% NOTE: mathematically this is wrong, because it replaces the mean of a
% function with the function of the mean. Nevertheless, I include it here
% to see how it behaves.
batch_size = 10;
online_reps = 5000;
mcmc_thin = 1;
stream = BinaryClassificationSampler(xs_true, ys_true, class_f, prior_var_w, zeros(2,1), 10*eye(2), mcmc_burnin, mcmc_thin);
step = LinearDecay(1, online_reps);
net_onln_derv = NeuralNet(2);
net_onln_derv.addLayer(ncomp, 'type', class_n, 'biases', zeros(ncomp, 1), 'fix_biases', true);
net_onln_derv.addLayer(1, 'type', 'linear', 'weights', ones(1, ncomp) / ncomp, 'fix_weights', true, 'biases', 0, 'fix_biases', true);
net_onln_derv.setLossFunction('avg_score_matching');
net_onln_derv.train_stream(stream, 'step', step, 'minibatch', batch_size, 'maxiter', online_reps);

% save results
save(fullfile('outdir', 'bags_of_samples', 'classification_demo_results.mat'));

%% -- plot everything
close all;

% options for the plots
xmin = -10; xmax = 10;
[X, Y] = meshgrid(linspace(xmin, xmax, 40));
wmin = -30; wmax = 30;
[Wx, Wy] = meshgrid(linspace(wmin, wmax, 100));
close all;

% plot unnormalized posterior on the weights and (some) weight samples
figure;
subplot(2,2,1); hold on;
perc_samples_to_plot = 0.1;
Z = reshape(log_post_w([Wx(:), Wy(:)]'), size(Wx));
idx = 1 : floor(1/perc_samples_to_plot) : ns_w_post;
surfc(Wx, Wy, exp(Z), 'facealpha', 0.2); shading flat; colorbar;
plot(ws_post(1, idx), ws_post(2, idx), 'y.');
plot(net_comp_derv.weights{1}(:,1), net_comp_derv.weights{1}(:,2), 'ko');
xlabel('w_1'); ylabel('w_2');
title(sprintf('p(w,D) and %d%% of MCMC samples', perc_samples_to_plot*100));
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);

% plot approximate posterior as computed by EP
subplot(2,2,2); hold on;
Z = reshape(log_gauss_pdf([Wx(:), Wy(:)]', m_ep, S_ep), size(Wx));
surfc(Wx, Wy, exp(Z), 'facealpha', 0.2); shading flat; colorbar;
xlabel('w_1'); ylabel('w_2');
title('Approximate posterior q(w) computed by EP');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([wmin wmax wmin wmax]);

% plot MCMC predictive model
subplot(2,2,3); hold on;
Z = reshape(net_post.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('MCMC predictive model');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);

% plot EP predictive model
subplot(2,2,4); hold on;
m_1d = [X(:), Y(:)] * m_ep;
S_1d = sum([X(:), Y(:)]' .* (S_ep * [X(:), Y(:)]'), 1);
Z = reshape(gauss_expectation(@(z) class_f(z), m_1d, S_1d), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('EP predictive model');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);

% plot compact predictive model
figure;
subplot(2,2,1); hold on;
Z = reshape(net_comp.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('Compact predictive model');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);

% plot compact predictive model fit using derivatives
subplot(2,2,2); hold on;
Z = reshape(net_comp_derv.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('Compact predictive model using derivatives');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);

% plot online predictive model
subplot(2,2,3); hold on;
Z = reshape(net_onln.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('Online predictive model');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);

% plot online predictive model fit using derivatives
subplot(2,2,4); hold on;
Z = reshape(net_onln_derv.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.2); shading flat; colorbar;
plot(xs_true(1, ys_true == 1), xs_true(2, ys_true == 1), 'r.', 'MarkerSize', 15);
plot(xs_true(1, ys_true ~= 1), xs_true(2, ys_true ~= 1), 'b.', 'MarkerSize', 15);
xlabel('x_1'); ylabel('x_2');
title('Online predictive model using derivatives');
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
