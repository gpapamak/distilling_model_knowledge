% Shows the results from compressing the binary net on mnist. Plots the
% training progress wrt (a) accuracy and (b) log probability on the test
% set. Also, it evaluates each network on each other's loss function.
%
% George Papamakarios, May 2015

clear;
outdir = fullfile('outdir', 'neural_nets', 'mnist_binary');

%% load mnist
load(fullfile('..', 'data', 'mnist', '2_vs_7.mat'));
%x_tst = x_trn;
%y_tst = y_trn;
num_trn = size(x_trn, 2);
num_tst = size(x_tst, 2);
clear x_trn y_trn;

%% training progress plots
close all;
sub = 3;
show_passes = 100;

% accuracy
figure(1); hold on;
title('accuracy');
xlabel('passes over train set');
xlim([0 show_passes]);
ylim([0.95 1]);

% log probability
figure(2); hold on;
title('log probability');
xlabel('passes over train set');
xlim([0 show_passes]);
ylim([-0.3 0]);

% net trained directly on data
load(fullfile(outdir, 'net_784_10_1.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'b');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'b');

% net trained with cross-entropy on large net
load(fullfile(outdir, 'compress_net_cross_entropy.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'r');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'r');

% net trained with average score matching on large net
load(fullfile(outdir, 'compress_net_avg_score_matching_nade.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'y');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'y');

% net trained with square error on logit on large net
load(fullfile(outdir, 'compress_net_square_error_logit.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'c');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'c');

% net trained with deriv square error on logit on large net
load(fullfile(outdir, 'compress_net_deriv_square_error_logit.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'g');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'g');

% net trained with square error & deriv sq error on logit on large net
load(fullfile(outdir, 'compress_net_square_error_&_deriv_square_error_1_logit.mat'));
t = batch_size * check_every / num_trn; xx = t : t : passes;
figure(1); plot(xx(1:sub:end), all_acc(1:sub:end), 'm');
figure(2); plot(xx(1:sub:end), all_logprob(1:sub:end), 'm');

% add legends to plots
figure(1); legend('train on data', 'cross entropy', 'avg score matching (with nade)', 'square error on logit', 'deriv square error on logit', 'square error & deriv square error on logit', 'Location', 'SouthEast');
figure(2); legend('train on data', 'cross entropy', 'avg score matching (with nade)', 'square error on logit', 'deriv square error on logit', 'square error & deriv square error on logit', 'Location', 'SouthEast');

%% evaluate nets on each other's loss functions

% load large net
load(fullfile(outdir, 'net_784_100_1.mat'));
large_net = net;

[y, dydx] = large_net.eval(x_tst);
dydx = [permute(y, [3 1 2]); dydx];

clear large_net net;

% net trained directly on data
load(fullfile(outdir, 'net_784_10_1.mat'));

net.setLossFunction('cross_entropy');
L1 = net.eval_loss(x_tst, y_tst);
L2 = net.eval_loss(x_tst, y);

net.setLossFunction('avg_score_matching');
L3 = net.eval_loss(x_tst, dydx);

fprintf('-- Net trained directly on data \n');
fprintf('Cross entropy vs test data      = %g \n', L1);
fprintf('Cross entropy vs large net      = %g \n', L2);
fprintf('Avg score matching vs large net = %g \n', L3);
fprintf('\n');

% net trained with cross-entropy on large net
load(fullfile(outdir, 'compress_net_cross_entropy.mat'));

net.setLossFunction('cross_entropy');
L1 = net.eval_loss(x_tst, y_tst);
L2 = net.eval_loss(x_tst, y);

net.setLossFunction('avg_score_matching');
L3 = net.eval_loss(x_tst, dydx);

fprintf('-- Net trained on large net by cross-entropy \n');
fprintf('Cross entropy vs test data      = %g \n', L1);
fprintf('Cross entropy vs large net      = %g \n', L2);
fprintf('Avg score matching vs large net = %g \n', L3);
fprintf('\n');

% net trained with gradient matching on large net
load(fullfile(outdir, 'compress_net_avg_score_matching.mat'));
%load(fullfile(outdir, 'compress_net_avg_score_matching_randn.mat'));

net.setLossFunction('cross_entropy');
L1 = net.eval_loss(x_tst, y_tst);
L2 = net.eval_loss(x_tst, y);

net.setLossFunction('avg_score_matching');
L3 = net.eval_loss(x_tst, dydx);

fprintf('-- Net trained on ensemble by avg score matching \n');
fprintf('Cross entropy vs test data      = %g \n', L1);
fprintf('Cross entropy vs large net      = %g \n', L2);
fprintf('Avg score matching vs large net = %g \n', L3);
fprintf('\n');

% NOTE: the other nets were not included in the above evaluation because of
% laziness...
