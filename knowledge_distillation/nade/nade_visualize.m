clear;

% load nade
num_nades = 60;
nades = cell(1, num_nades);

for i = 1:num_nades
    load(fullfile('outdir', 'nade', 'ensemble', sprintf('mnist_allclass_500_logistic_%d.mat', i)), 'nade');
    nades{i} = nade;
end

nade = NadeEnsemble(nades);
clear num_nades nades;

% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_tst');
x_tst = double(x_tst > 0.5);

%% visualize
close all;
pixel = 1:10:nade.num_inputs;
range = linspace(0, 1, 10);

for i = 1:numel(pixel)
    x = x_tst(:, 1) * ones(size(range));
    x(pixel(i), :) = range;
    L = nade.eval(x);

    figure;
    plot(range, L);
    title(sprintf('pixel = %d, value = %d', pixel(i), x_tst(pixel(i), 1)));
end
