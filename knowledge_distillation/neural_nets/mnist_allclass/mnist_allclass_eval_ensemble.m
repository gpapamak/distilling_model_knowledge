% Evaluates the performance on the mnist test set of (a) the neural
% ensemble, (b) each component of the ensemble and (c) a single net trained
% on mnist without bagging of the same size as each component.
%
% George Papamakarios, May 2015

clear;
close all;
outdir = fullfile('outdir', 'neural_nets', 'mnist_allclass');

%% load ensemble
num_nets = 30;
nets = cell(1, num_nets);

for i = 1:num_nets
    
    load(fullfile(outdir, 'ensemble', sprintf('net_%d.mat', i)));
    net.changeLayerType('final', 'softmax');
    nets{i} = net;
end
ensemble = NeuralEnsemble(nets);

clearvars -except ensemble outdir;

%% load single net
load(fullfile(outdir, 'net_784_500_300_10.mat'));
net.changeLayerType('final', 'softmax');

clearvars -except ensemble net;

%% load mnist
load(fullfile('data', 'mnist', 'all_digits.mat'));
%x_tst = x_trn;
%y_tst = y_trn;
num_tst = size(x_tst, 2);
[~, y_tst_label] = max(y_tst);
y_tst_label = y_tst_label - 1;

%% evaluate performance

% evaluate each of the ensemble components
for i = 1:ensemble.num_nets
    y = ensemble.nets{i}.eval(x_tst);
    [~, y_prd] = max(y);
    y_prd = y_prd - 1;
    acc = y_prd == y_tst_label;
    acc_avg = mean(acc);
    acc_std = std(acc) / sqrt(num_tst);
    logprob = log(y(logical(y_tst)));
    logprob_avg = mean(logprob);
    logprob_std = std(logprob) / sqrt(num_tst);
    fprintf('Net = %d, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', i, acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);
end

% evaluate the whole ensemble
y = ensemble.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_label;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = log(y(logical(y_tst)));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('Ensemble, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);

% evaluate the single net
y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_label;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = log(y(logical(y_tst)));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('Single net, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);
