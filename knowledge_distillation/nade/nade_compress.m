clear;
close all;

platform = 'gpu';
outdir = fullfile('outdir', 'nade');

% make the arrangements for the platform
switch platform
    case 'cpu'
        createArray = @(x) x;
    case 'gpu'
        createArray = @gpuArray;
    otherwise
        error('Uknown platform.');
end

% load samples from the nade to compress
load(fullfile(outdir, 'ensemble', 'samples_ensemble.mat'), 'x', 'L', 'dLdx');
num_samples = 100000;
x = createArray(x(:, 1:num_samples));
L = createArray(L(:, 1:num_samples));
dLdx = createArray(dLdx(:, 1:num_samples));

% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_tst');
x_tst = createArray(x_tst(:, 1:500));
x_tst = double(x_tst > 0.5);

% train parameters
batch_size = 20;
maxiter = 30000;
monitor_every = 200;
type = 'logistic';
num_hidden = 500;
loss_fun = 'max_likelihood';

% train nade
nade = Nade(size(x, 1), num_hidden, type, platform);
stream = DataSubSampler(x, L, dLdx);
progress = nade.train_stream(stream, 'loss', loss_fun, 'minibatch', batch_size, 'maxiter', maxiter, 'monitor_every', monitor_every, 'x_tst', x_tst);

% save nade and training progress
outfile = sprintf('compress_ensemble_%d_%s_%s.mat', num_hidden, type, loss_fun);
nade.changePlatform('cpu');
save(fullfile(outdir, outfile), 'loss_fun', 'nade', 'progress', 'maxiter', 'batch_size', 'monitor_every');
