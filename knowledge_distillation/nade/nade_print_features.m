% Prints features of the rbm trained on binarized mnist, and of the various
% nades that were trained to mimic it.
%
% George Papamakarios, Jul 2015

clear;
close all;

outdir = fullfile('outdir', 'rbm');
savedir = fullfile('..', 'reports', 'figs', 'generative_models');

D1 = 28;
D2 = 28;

N1 = 5;
N2 = 8;

range = [];

%% -- rbm

load(fullfile('data', 'rbm', 'rbm_CD25_500.mat'), 'rbm');

features = zeros(D1*N1, D2*N2);

n = 0;
for n1 = 1:N1
    for n2 = 1:N2
        n = n + 1;
        ii = (n1-1)*D1+1 : n1*D1;
        jj = (n2-1)*D2+1 : n2*D2;
        features(ii,jj) = reshape(rbm.W(:,n), [D1,D2]);
    end
end

fig = figure;
imshow(features, range);
filename = fullfile(savedir, 'rbm_features.eps');
saveTightFigure(fig, filename);
system(['epstopdf ', filename]);
system(['rm -f ', filename]);

%% -- nade

num_hidden = [1000, 750, 500, 250];

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_max_likelihood.mat', i)), 'nade');
    
    features = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            features(ii,jj) = reshape([nade.W(:,n); 0], [D1,D2]);
        end
    end

    fig = figure;
    imshow(features, range);
    filename = fullfile(savedir, sprintf('nade_kl_divergence_%d_features.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
end

for i = num_hidden
    
    load(fullfile(outdir, sprintf('mimic_rbm_%d_logistic_square_error_c436.49.mat', i)), 'nade');
    
    features = zeros(D1*N1, D2*N2);

    n = 0;
    for n1 = 1:N1
        for n2 = 1:N2
            n = n + 1;
            ii = (n1-1)*D1+1 : n1*D1;
            jj = (n2-1)*D2+1 : n2*D2;
            features(ii,jj) = reshape([nade.W(:,n); 0], [D1,D2]);
        end
    end

    fig = figure;
    imshow(features, range);
    filename = fullfile(savedir, sprintf('nade_square_error_%d_features.eps', i));
    saveTightFigure(fig, filename);
    system(['epstopdf ', filename]);
    system(['rm -f ', filename]);
end
