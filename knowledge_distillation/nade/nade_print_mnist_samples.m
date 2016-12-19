% Generates and prints samples from binarized mnist, the rbm trained on it
% and the various nades that were trained to mimic it.
%
% George Papamakarios, Jul 2015

clear;
close all;

outdir = fullfile('outdir', 'rbm');
savedir = fullfile('..', 'reports', 'figs', 'generative_models');

D1 = 28;
D2 = 28;
D = D1 * D2;

N1 = 5;
N2 = 8;
N = N1 * N2;

%% -- mnist

load(fullfile('data', 'mnist', 'randomly_binarized_mnist.mat'), 'x_trn');
x = data_sample(x_trn, false, N);
samples = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        samples(ii,jj) = reshape(x(:,n), [D1,D2]);
    end
end

fig = figure;
imshow(samples);
filename = fullfile(savedir, 'mnist_samples.eps');
saveTightFigure(fig, filename);
system(['epstopdf ', filename]);
system(['rm -f ', filename]);

%% -- rbm

load(fullfile('data', 'rbm', 'rbm_CD25_500.mat'), 'rbm');

rbm.setGibbsState(double(rand(D, N) > 0.5));
x = rbm.gen(N, 2000);
samples = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        samples(ii,jj) = reshape(x(:,n), [D1,D2]);
    end
end

fig = figure;
imshow(samples);
filename = fullfile(savedir, 'rbm_samples.eps');
saveTightFigure(fig, filename);
system(['epstopdf ', filename]);
system(['rm -f ', filename]);

%% -- nade

num_hidden = [1000, 750, 500, 250];

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    
    [~, x] = nade.gen(N);
    samples = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            samples(ii,jj) = reshape(x(:,n), [D1,D2]);
        end
    end

    fig = figure;
    imshow(samples);
    filename = fullfile(savedir, sprintf('nade_kl_divergence_%d_samples.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
end

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error_c436.49.mat', i)), 'nade');
    
    [~, x] = nade.gen(N);
    samples = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            samples(ii,jj) = reshape(x(:,n), [D1,D2]);
        end
    end

    fig = figure;
    imshow(samples);
    filename = fullfile(savedir, sprintf('nade_square_error_%d_samples.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
end
