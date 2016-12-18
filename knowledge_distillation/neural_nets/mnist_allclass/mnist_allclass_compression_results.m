% Shows the results from compressing the neural ensemble on mnist. Plots
% the training progress wrt (a) accuracy and (b) log probability on the
% test set.
%
% George Papamakarios, May 2015

clear;
close all;
clc;

outdir = fullfile('outdir', 'neural_nets', 'mnist_allclass');
col = distinguishable_colors(3);
num_trn = 6000;
xcrop = [2000, 40*num_trn];
ycrop_acc = 100 * [0.3, 1];
ycrop_logprob = [-2, 0];
plotfun_acc = @semilogx;
plotfun_logprob = @semilogx;
fontsize = 18;
linewidth = 3;
num_stds = 2;
savedir = fullfile('..', 'reports', 'figs', 'model_compression');

%% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_tst', 'y_tst');
num_tst = size(x_tst, 2);
[~, y_tst_lb] = max(y_tst);
y_tst_lb = y_tst_lb - 1;

%% ensemble
num_nets = 30;
nets = cell(1, num_nets);

for i = 1:num_nets
    load(fullfile(outdir, 'ensemble', sprintf('net_%d.mat', i)), 'net');
    nets{i} = net;
end
ensemble = NeuralEnsemble(nets, 'logmeanexp');
clear num_nets net nets;

y = ensemble.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('ensemble \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100);
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);
fprintf('\n');
ensemble_acc = acc_avg;
ensemble_logprob = logprob_avg;

%% net trained on data
load(fullfile(outdir, sprintf('net_784_50_30_10_%dk.mat', num_trn/1000)), 'net');

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('net trained on data \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100);
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);
fprintf('\n');
net_on_data_acc = acc;
net_on_data_logprob = logprob;

%% cross entropy

fprintf('** cross entropy ** \n');

figure(1); plotfun_acc(xcrop, 100*ensemble_acc*[1 1], '--', 'Color', [139, 149, 173]/255, 'LineWidth', linewidth); hold on;
figure(2); plotfun_logprob(xcrop, ensemble_logprob*[1 1], '--', 'Color', [139, 149, 173]/255, 'LineWidth', linewidth); hold on;

% mnist
load(fullfile(outdir, sprintf('compress_ensemble_cross_entropy_mnist_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(1); plotfun_acc(xx, 100*all_acc, 'Color', col(1,:), 'LineWidth', linewidth);
figure(2); plotfun_logprob(xx, all_logprob, 'Color', col(1,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('mnist \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% nade
load(fullfile(outdir, sprintf('compress_ensemble_cross_entropy_nade_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(1); plotfun_acc(xx, 100*all_acc, 'Color', col(2,:), 'LineWidth', linewidth);
figure(2); plotfun_logprob(xx, all_logprob, 'Color', col(2,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('nade \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% randn
load(fullfile(outdir, sprintf('compress_ensemble_cross_entropy_randn_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(1); plotfun_acc(xx, 100*all_acc, 'Color', col(3,:), 'LineWidth', linewidth);
figure(2); plotfun_logprob(xx, all_logprob, 'Color', col(3,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('randn \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% decorate plots
figure(1);
%title('Cross entropy', 'FontSize', fontsize);
xlabel('Number of train samples', 'FontSize', fontsize);
ylabel('Accuracy [%]', 'FontSize', fontsize);
xlim(xcrop); ylim(ycrop_acc);
legend('Ensemble', 'MNIST', 'NADE', 'Random', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
filename = fullfile(savedir, sprintf('compress_ensemble_cross_entropy_accuracy_%dk', num_trn/1000));
print([filename, '.eps'], '-depsc');
system(['epstopdf ', filename, '.eps']);
system(['rm -f ', filename, '.eps']);

figure(2);
%title('Cross entropy', 'FontSize', fontsize);
xlabel('Number of train samples', 'FontSize', fontsize);
ylabel('Average log probability', 'FontSize', fontsize);
xlim(xcrop); ylim(ycrop_logprob);
legend('Ensemble', 'MNIST', 'NADE', 'Random', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
filename = fullfile(savedir, sprintf('compress_ensemble_cross_entropy_logprob_%dk', num_trn/1000));
print([filename, '.eps'], '-depsc');
system(['epstopdf ', filename, '.eps']);
system(['rm -f ', filename, '.eps']);

fprintf('\n');

%% derivative square error

fprintf('** derivative square error ** \n');

figure(3); plotfun_acc(xcrop, 100*ensemble_acc*[1 1], '--', 'Color', [139, 149, 173]/255, 'LineWidth', linewidth); hold on;
figure(4); plotfun_logprob(xcrop, ensemble_logprob*[1 1], '--', 'Color', [139, 149, 173]/255, 'LineWidth', linewidth); hold on;

% mnist
load(fullfile(outdir, sprintf('compress_ensemble_deriv_square_error_mnist_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(3); plotfun_acc(xx, 100*all_acc, 'Color', col(1,:), 'LineWidth', linewidth);
figure(4); plotfun_logprob(xx, all_logprob, 'Color', col(1,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('mnist \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% nade
load(fullfile(outdir, sprintf('compress_ensemble_deriv_square_error_nade_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(3); plotfun_acc(xx, 100*all_acc, 'Color', col(2,:), 'LineWidth', linewidth);
figure(4); plotfun_logprob(xx, all_logprob, 'Color', col(2,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('nade \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% randn
load(fullfile(outdir, sprintf('compress_ensemble_deriv_square_error_randn_%dk.mat', num_trn/1000)));
xx = check_every * batch_size : check_every * batch_size : passes * num_trn;
figure(3); plotfun_acc(xx, 100*all_acc, 'Color', col(3,:), 'LineWidth', linewidth);
figure(4); plotfun_logprob(xx, all_logprob, 'Color', col(3,:), 'LineWidth', linewidth);

y = net.eval(x_tst);
[~, y_prd] = max(y);
y_prd = y_prd - 1;
acc = y_prd == y_tst_lb;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(num_tst);
logprob = y(logical(y_tst));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(num_tst);
fprintf('randn \n');
fprintf('  accuracy = $%.2f \\pm %.2f$ \n', acc_avg*100, num_stds*acc_std*100); 
fprintf('  log-prob = $%.3f \\pm %.3f$ \n', logprob_avg, num_stds*logprob_std);

diff_acc = acc - net_on_data_acc;
diff_logprob = logprob - net_on_data_logprob;
fprintf('  accuracy diff = $%.2f \\pm %.2f$ \n', mean(diff_acc)*100, num_stds*std(diff_acc)/sqrt(num_tst)*100);
fprintf('  log-prob diff = $%.3f \\pm %.3f$ \n', mean(diff_logprob), num_stds*std(diff_logprob)/sqrt(num_tst));

% decorate plots
figure(3);
%title('Derivative square error', 'FontSize', fontsize);
xlabel('Number of train samples', 'FontSize', fontsize);
ylabel('Accuracy [%]', 'FontSize', fontsize);
xlim(xcrop); ylim(ycrop_acc);
legend('Ensemble', 'MNIST', 'NADE', 'Random', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
filename = fullfile(savedir, sprintf('compress_ensemble_deriv_square_error_accuracy_%dk', num_trn/1000));
print([filename, '.eps'], '-depsc');
system(['epstopdf ', filename, '.eps']);
system(['rm -f ', filename, '.eps']);

figure(4);
%title('Derivative square error', 'FontSize', fontsize);
xlabel('Number of train samples', 'FontSize', fontsize);
ylabel('Average log probability', 'FontSize', fontsize);
xlim(xcrop); ylim(ycrop_logprob);
legend('Ensemble', 'MNIST', 'NADE', 'Random', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
filename = fullfile(savedir, sprintf('compress_ensemble_deriv_square_error_logprob_%dk', num_trn/1000));
print([filename, '.eps'], '-depsc');
system(['epstopdf ', filename, '.eps']);
system(['rm -f ', filename, '.eps']);

fprintf('\n');
