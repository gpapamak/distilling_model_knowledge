% Trains a nade to mimic an rbm.
%
% George Papamakarios, Jul 2015

clear;
close all;

platform = 'gpu';
outdir = fullfile('outdir', 'rbm');

% make the arrangements for the platform
switch platform
    case 'cpu'
        createArray = @(x) x;
    case 'gpu'
        createArray = @gpuArray;
    otherwise
        error('Unknown platform.');
end

% load mnist
% note that this is Salakhutdinov and Murray's binarization; I need to use
% it here because the rbm was trained on data binarized this way
load(fullfile(outdir, 'randomly_binarized_mnist.mat'), 'x_trn', 'x_tst');
x_trn = createArray(x_trn);
x_tst = createArray(x_tst(:, 1:500));

% load rbm
load(fullfile(outdir, 'rbm_CD25_500.mat'));
rbm.changePlatform(platform);
[L, dLdx] = rbm.eval(x_trn);

% train parameters
batch_size = 20;
maxiter = 30000;
monitor_every = 200;
type = 'logistic';
num_hidden = 500;
loss_fun = {'max_likelihood', 'square_error'};
burnin = 2000;
num_chains = 100 * batch_size;

for i = 1:numel(loss_fun)
    
    % initialize rbm stream
    rbm.setGibbsState(double(rand(rbm.num_inputs, num_chains) > 0.5));
    stream = RbmStream(rbm, burnin);
    %stream = DataSubSampler(x_trn, L, dLdx);

    % train nade
    nade = Nade(rbm.num_inputs, num_hidden, type, platform, 1:rbm.num_inputs, 'single');
    progress = nade.train_stream(stream, 'loss', loss_fun{i}, 'minibatch', batch_size, 'maxiter', maxiter, 'monitor_every', monitor_every, 'x_tst', x_tst);
    
    % save nade and training progress
    outfile = sprintf('mimic_rbm_%d_%s_%s.mat', num_hidden, type, loss_fun{i});
    nade.changePlatform('cpu');
    save(fullfile(outdir, outfile), 'nade', 'progress', 'maxiter', 'batch_size', 'monitor_every', 'burnin', 'num_chains');
end
