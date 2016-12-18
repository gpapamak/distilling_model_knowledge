% Density estimation of a mixture of 1d gaussians by fitting a compact 
% representation to the bayesian predictive distribution.
% Based on: 
% E. Snelson and Z. Ghahramani, "Compact approximations to
% Bayesian predictive distributions," ICML, 2005.
%
% George Papamakarios, Jan 2015

clear;
close all;

% true parameters
a_true = [1 1 1]' / 3;
m_true = [-3 0 2];
s_true = [2 5 1];
K = length(a_true);

% generate samples from the true pdf
ns_y_true = 10;
ys_true = mog_sample(a_true, m_true, s_true, ns_y_true);

%% -- fit models

% generate samples from the posterior on the weights
prior_var_m = 100;
ns_m_post = 1e+4;
mcmc_burnin = floor(0.1 * ns_m_post);
log_post_m = @(m) sum(log_mog_pdf(ys_true, a_true, m, s_true)) - m(:)'*m(:) / (2*prior_var_m);
ms_post = slice_sample(log_post_m, zeros(1,K), ns_m_post, 'burnin', mcmc_burnin);

% generate samples from the approximate predictive pdf 
% and fit a compact mog on them using em
ns_y_pred = 1e+3;
a_pred = repmat(a_true, ns_m_post, 1) / ns_m_post;
m_pred = ms_post(:)';
s_pred = repmat(s_true, 1, ns_m_post);
ys_pred = mog_sample(a_pred, m_pred, s_pred, ns_y_pred);
[a_comp, m_comp, s_comp] = em_mog(ys_pred, K, 'verbose', true);

% fit a compact mog online
batch_size = 100;
online_reps = 100;
thinning = 1;
[ms_post_batch, mcmc_width] = slice_sample(log_post_m, zeros(1,K), 'burnin', mcmc_burnin);
a_onln = ones(K, 1) / K;
m_onln = randn(1, K);
s_onln = ones(1, 1, K);

for i = 1:online_reps

    % generate a new batch
    ms_post_batch = slice_sample(log_post_m, ms_post_batch(:,end)', batch_size, 'thin', thinning, 'width', mcmc_width);
    ys_pred = zeros(1, batch_size);
    for j = 1:batch_size
        ys_pred(j) = mog_sample(a_true, ms_post_batch(:,j)', s_true);
    end

    % em update
    step = 1 - (i - 1) / (online_reps - 1);
    %step = 1 / (2 + i) ^ 1.0;
    [a_onln, m_onln, s_onln] = em_mog_stepwise(ys_pred, a_onln, m_onln, s_onln, step);

    i
end

% save results
save(fullfile('outdir', 'bags_of_samples', sprintf('density_demo_results_%d.mat', ns_y_true)));

%% -- plot everything
close all;

% plot approximate posterior marginals on the weights
figure;
suptitle('posterior marginals using MCMC');
for k = 1:K
    subplot(1, K, k);
    [mm, marg_post_m] = samples2pdf(ms_post(k,:));
    plot(mm, marg_post_m, 'r'); 
    xlabel(sprintf('m_%d', k));
end

% plot true, predictive and compact pdfs
figure; hold on;
yy = linspace(-12, 8);
pdf_true = exp(log_mog_pdf(yy, a_true, m_true, s_true));
pdf_pred = exp(log_mog_pdf(yy, a_pred, m_pred, s_pred));
pdf_comp = exp(log_mog_pdf(yy, a_comp, m_comp, s_comp));
pdf_onln = exp(log_mog_pdf(yy, a_onln, m_onln, s_onln));
plot(yy, pdf_true, 'b');
plot(yy, pdf_pred, 'g');
plot(yy, pdf_comp, 'r');
plot(yy, pdf_onln, 'k');
plot(ys_true, 0, 'ro', 'MarkerFaceColor', 'r');
xlabel('y');
legend('true', 'bayesian', 'compact', 'online', 'samples', 'Location', 'NorthWest');
