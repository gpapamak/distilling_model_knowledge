clear;
close all;

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

% load ensemble
num_nades = 60;
nades = cell(1, num_nades);

for i = 1:num_nades
    load(fullfile('outdir', 'nade', 'ensemble', sprintf('mnist_allclass_500_logistic_%d.mat', i)), 'nade');
    nades{i} = nade;
end

ensemble = NadeEnsemble(nades, platform);
clear nade num_nades nades;

% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_tst');
num_tst = 500;
x_tst = createArray(x_tst(:, 1:num_tst));
x_tst = double(x_tst > 0.5);

% evaluate each of the ensemble components
for i = 1:ensemble.num_nades
    L = ensemble.nades{i}.eval(x_tst);
    L_avg = mean(L);
    L_std = std(L) / sqrt(num_tst);
    fprintf('Nade = %2d, log-probability = %g +- %g \n', i, L_avg, 2*L_std);
end

% evaluate the whole ensemble
L = ensemble.eval(x_tst);
L_avg = mean(L);
L_std = std(L) / sqrt(num_tst);
fprintf('Ensemble, log-probability = %g +- %g \n', L_avg, 2*L_std);
