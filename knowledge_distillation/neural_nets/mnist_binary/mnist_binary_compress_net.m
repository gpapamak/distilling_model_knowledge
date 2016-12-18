% Compress a large neural net trained to differentiate between two digits
% to a small neural net.
%
% George Papamakarios, May 2015

clear;
close all;
outdir = fullfile('outdir', 'neural_nets', 'mnist_binary');

% select type of compression to use
%loss_function = 'cross_entropy'; outfile = 'compress_net_cross_entropy.mat';
loss_function = 'avg_score_matching'; outfile = 'compress_net_avg_score_matching_nade.mat';

%% load large net
load(fullfile(outdir, 'net_784_100_1.mat'));
large_net = net;
clearvars -except large_net loss_function outfile outdir;

%% load mnist
load(fullfile('..', 'data', 'mnist', '2_vs_7.mat'));
num_trn = size(x_trn, 2);
num_tst = size(x_tst, 2);
clear y_trn;

%% load nade
load(fullfile('outdir', 'nade', 'mnist_binary_100.mat'));

%% compress large net
net = NeuralNet(large_net.num_inputs);
net.addLayer(10, 'type', 'logistic');
net.addLayer( 1, 'type', 'logistic');
net.setLossFunction(loss_function);

% train the net
passes = 100;
batch_size = 20;
num_epochs = passes * floor(num_trn / batch_size);
check_every = 100;
all_acc = [];
all_logprob = [];
stream = NetworkSampler(large_net, @nade.gen);
%stream = NetworkSampler(large_net, x_trn);
step = AdaDelta();

for i = 1:floor(num_epochs / check_every)
    
    net.train_stream(stream, 'step', step, 'minibatch', batch_size, 'maxiter', check_every, 'verbose', false);
    
    % performance on the test set
    y = net.eval(x_tst);
    acc = round(y) == y_tst;
    acc_avg = mean(acc);
    acc_std = std(acc) / sqrt(num_tst);
    logprob = y_tst .* log(y) + (1-y_tst) .* log(1-y);
    logprob_avg = mean(logprob);
    logprob_std = std(logprob) / sqrt(num_tst);
    fprintf('Iteration = %d, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', i, acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);
    all_acc = [all_acc, acc_avg]; %#ok<AGROW>
    all_logprob = [all_logprob, logprob_avg]; %#ok<AGROW>
end

%% save the net and training progress
save(fullfile(outdir, outfile), 'net', 'all_acc', 'all_logprob', 'passes', 'batch_size', 'check_every');
