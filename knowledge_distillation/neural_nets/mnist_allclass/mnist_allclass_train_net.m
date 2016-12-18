% Trains a net on mnist with all digit classes and softmax output. Also
% monitors the training progress. Saves both the net and the training
% progress.
%
% George Papamakarios, May 2015

clear;
close all;
outdir = fullfile('outdir', 'neural_nets', 'mnist_allclass');
platform = 'gpu';

% make the arrangements for the platform
switch platform
    case 'cpu'
        createArray = @(x) x;
    case 'gpu'
        createArray = @gpuArray;
    otherwise
        error('Uknown platform.');
end

% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_trn', 'x_tst', 'y_trn', 'y_tst');
num_trn = 60000;
x_trn = createArray(x_trn(:, 1:num_trn));
x_tst = createArray(x_tst);
y_trn = createArray(y_trn(:, 1:num_trn));
y_tst = createArray(y_tst);
num_inputs = size(x_trn, 1);
num_tst = size(x_tst, 2);
[~, y_tst_lb] = max(y_tst);
y_tst_lb = y_tst_lb - 1;

% create the net
net = NeuralNet(num_inputs, platform);
net.addLayer(50, 'type', 'relu');
net.addLayer(30, 'type', 'relu');
net.addLayer(10, 'type', 'logsoftmax');
net.setLossFunction('dot_product');

% train the net
passes = 20;
batch_size = 20;
num_epochs = passes * floor(num_trn / batch_size);
check_every = 100;
all_acc = [];
all_logprob = [];
stream = DataSubSampler(x_trn, y_trn);
step = AdaDelta();

for i = 1:floor(num_epochs / check_every)
    
    net.train_stream(stream, 'step', step, 'minibatch', batch_size, 'maxiter', check_every, 'verbose', false);
        
    % performance on the test set
    y = net.eval(x_tst);
    [~, y_prd] = max(y);
    y_prd = y_prd - 1;
    acc = y_prd == y_tst_lb;
    acc_avg = mean(acc);
    acc_std = std(acc) / sqrt(num_tst);
    logprob = y(logical(y_tst));
    logprob_avg = mean(logprob);
    logprob_std = std(logprob) / sqrt(num_tst);
    fprintf('Iteration = %d, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', i, acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);
    all_acc = [all_acc, acc_avg]; %#ok<AGROW>
    all_logprob = [all_logprob, logprob_avg]; %#ok<AGROW>
end

% save the net and training progress
net.changePlatform('cpu');
all_acc = gather(all_acc);
all_logprob = gather(all_logprob);
save(fullfile(outdir, sprintf('net_784_50_30_10_%dk.mat', num_trn/1000)), 'net', 'all_acc', 'all_logprob', 'passes', 'batch_size', 'check_every');
