clear;
close all;

outdir = fullfile('outdir', 'nade', 'compress', '100000_samples');
num_trn = 60000;
num_hidden = 500;
reg = [0.001, 0.01, 0.1, 1, 10, 100, 1000];

close all;
sub = 1;

%% net trained on ensemble with max likelihood regularized by score matching

final_ml_sm = zeros(1, numel(reg));
best_ml_sm = [];
best = -inf;

for i = 1:numel(reg)
    
    load(fullfile(outdir, sprintf('compress_ensemble_%d_logistic_max_likelihood_&_score_matching_%g.mat', num_hidden, reg(i))));
    final_ml_sm(i) = progress.tst(end);
    
    if final_ml_sm(i) > best
        best_ml_sm = progress.tst;
        best = final_ml_sm(i);
    end
end

%% net trained on ensemble with square error regularized by score matching

final_se_sm = zeros(1, numel(reg));
best_se_sm = [];
best = -inf;

for i = 1:numel(reg)
    
    load(fullfile(outdir, sprintf('compress_ensemble_%d_logistic_square_error_&_score_matching_%g.mat', num_hidden, reg(i))));
    final_se_sm(i) = progress.tst(end);
    
    if final_se_sm(i) > best
        best_se_sm = progress.tst;
        best = final_se_sm(i);
    end
end

%% no regularization

figure;

% net trained directly on data
load(fullfile('outdir', 'nade', sprintf('mnist_allclass_%d_logistic.mat', num_hidden)));
t = batch_size * monitor_every / num_trn; xx = t : t : passes; xx = [0 xx];
plot(xx(1:sub:end), progress.tst(1:sub:end), 'g'); hold on;
final_dt = progress.tst(end);

% net trained on ensemble with max likelihood
load(fullfile(outdir, sprintf('compress_ensemble_%d_logistic_max_likelihood.mat', num_hidden)));
t = batch_size * monitor_every / num_trn; xx = t : t : passes; xx = [0 xx];
plot(xx(1:sub:end), progress.tst(1:sub:end), 'c'); hold on;
final_ml = progress.tst(end);

% max likelihood regularized with score matching
plot(xx(1:sub:end), best_ml_sm(1:sub:end), 'b'); hold on;

% net trained on ensemble with square error
load(fullfile(outdir, sprintf('compress_ensemble_%d_logistic_square_error.mat', num_hidden)));
t = batch_size * monitor_every / num_trn; xx = t : t : passes; xx = [0 xx];
plot(xx(1:sub:end), progress.tst(1:sub:end), 'm'); hold on;
final_se = progress.tst(end);

% square error regularized with score matching
plot(xx(1:sub:end), best_se_sm(1:sub:end), 'r'); hold on;

% net trained on ensemble with score matching
load(fullfile(outdir, sprintf('compress_ensemble_%d_logistic_score_matching.mat', num_hidden)));
t = batch_size * monitor_every / num_trn; xx = t : t : passes; xx = [0 xx];
plot(xx(1:sub:end), progress.tst(1:sub:end), 'y'); hold on;
final_sm = progress.tst(end);

title('log probability');
xlabel('passes over train set');
legend('train on data', 'max likelihood', 'max likelihood & score matching', 'square error', 'square error & score matching', 'score matching', 'Location', 'SouthEast');

%% final values only

figure;
semilogx([min(reg) max(reg)], final_dt * [1 1], 'g'); hold on;
semilogx([min(reg) max(reg)], final_ml * [1 1], 'c');
semilogx(reg, final_ml_sm, 'bo:');
semilogx([min(reg) max(reg)], final_se * [1 1], 'm');
semilogx(reg, final_se_sm, 'ro:');
xlabel('regularizer');
title('final log probability');
legend('train on data', 'maximum likelihood', 'maximum likelihood & score matching', 'square error', 'square error & score matching', 'Location', 'SouthWest');
