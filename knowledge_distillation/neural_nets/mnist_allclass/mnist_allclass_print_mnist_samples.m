% Generates and prints samples from mnist and the nades that were trained
% on it.
%
% George Papamakarios, Jul 2015

clear;
close all;

savedir = fullfile('..', 'reports', 'figs', 'model_compression');

D1 = 28;
D2 = 28;
D = D1 * D2;

N1 = 5;
N2 = 8;
N = N1 * N2;

%% -- mnist

load(fullfile('data', 'mnist', 'all_digits.mat'), 'x_trn');
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

fig = figure;
imshow(double(samples > 0.5));
filename = fullfile(savedir, 'mnist_samples_binary.eps');
saveTightFigure(fig, filename);
system(['epstopdf ', filename]);
system(['rm -f ', filename]);

%% -- nade

num_trn = [6, 60];

for i = num_trn
    
    load(fullfile('outdir', 'nade', sprintf('mnist_allclass_500_logistic_%dk.mat', i)), 'nade');
    
    [y, x] = nade.gen(N);
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
    filename = fullfile(savedir, sprintf('nade_samples_binary_%dk.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
    
    samples = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            samples(ii,jj) = reshape(y(:,n), [D1,D2]);
        end
    end

    fig = figure;
    imshow(samples);
    filename = fullfile(savedir, sprintf('nade_conditional_prob_%dk.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
end
