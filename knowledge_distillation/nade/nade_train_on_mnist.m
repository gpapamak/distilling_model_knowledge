clear;
close all;
rng('shuffle');

platform = 'gpu';

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
load(fullfile('data', 'mnist', 'all_digits.mat'), 'x_trn', 'x_tst');
x_trn = createArray(x_trn);
x_tst = createArray(x_tst(:, 1:500));
x_trn = double(x_trn > 0.5);
x_tst = double(x_tst > 0.5);
[num_inputs, num_trn] = size(x_trn);

% train params
passes = 10;
batch_size = 20;
maxiter = passes * floor(num_trn / batch_size);
monitor_every = 200;
type = 'logistic';
num_hidden = 500;

for i = 1:60

    % train nade
    nade = Nade(num_inputs, num_hidden, type, platform);
    progress = nade.train(x_trn, 'minibatch', batch_size, 'maxiter', maxiter, 'monitor_every', monitor_every, 'x_tst', x_tst);

    % save nade and training progress
    outdir = fullfile('outdir', 'nade', 'ensemble');
    outfile = sprintf('mnist_allclass_%d_%s_%d.mat', num_hidden, type, i);
    nade.changePlatform('cpu');
    save(fullfile(outdir, outfile), 'nade', 'progress', 'passes', 'batch_size', 'monitor_every');

end
