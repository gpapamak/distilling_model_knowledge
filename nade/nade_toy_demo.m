% A toy demo with nade.
%
% George Papamakarios, 2015

clear;
close all;

%% train a nade

% create data
N = 1000;
D = 100;
f = 0.1;
x = double(rand(D,N) < f);

% train nade
nade = Nade(D, 200, 'logistic');
maxiter = 500;
minibatch = 20;
progress = nade.train(x, 'maxiter', maxiter, 'minibatch', minibatch, 'monitor_every', 1);

% plot the learning progress
figure;
loglog(progress.trn);
xlabel('iterations');
ylabel('loss');

% generate from nade
figure;
S = 1000;
[~, x] = nade.gen(S);
hist(sum(x));

%% now compress it

% compress nade
nade_comp = Nade(D, 20, 'logistic');
maxiter = 500;
minibatch = 20;
progress = nade_comp.train_nade(nade, 'loss', 'score_matching', 'maxiter', maxiter, 'minibatch', minibatch, 'monitor_every', 1);

% plot the learning progress
figure;
loglog(progress.trn);
xlabel('iterations');
ylabel('loss');

% generate from nade
figure;
S = 1000;
[~, x] = nade_comp.gen(S);
hist(sum(x));
