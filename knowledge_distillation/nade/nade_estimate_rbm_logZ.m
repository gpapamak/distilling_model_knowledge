% Estimates the partition function of the RBM using importance/bridge
% sampling from the trained NADEs.
%
% George Papamakarios, Jul 2015

clear;
close all;

outdir = fullfile('outdir', 'rbm');
num_hidden = [250, 500, 750, 1000];
method = 'bridge_sampling';
num_samples = 10000;
platform = 'gpu';

% load rbm
load(fullfile(outdir, 'rbm_CD25_500.mat'), 'rbm');
rbm.changePlatform(platform);

% kl divergence
fprintf('** kl divergence ** \n');
for i = num_hidden
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    nade.changePlatform(platform);
    [logZ, conf] = rbm.estimate_logZ(nade, method, num_samples);
    fprintf('%d hiddens, logZ = %.2f conf = [%.2f, %.2f] \n', i, logZ, conf);
end
fprintf('\n');

% square error
fprintf('** square error ** \n');
for i = num_hidden
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error_c436.49.mat', i)), 'nade');
    nade.changePlatform(platform);
    [logZ, conf] = rbm.estimate_logZ(nade, method, num_samples);
    fprintf('%d hiddens, logZ = %.2f conf = [%.2f, %.2f] \n', i, logZ, conf);
end
fprintf('\n');
