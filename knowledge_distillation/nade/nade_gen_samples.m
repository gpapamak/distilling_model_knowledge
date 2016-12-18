clear;
rng('shuffle');

platform = 'gpu';
outdir = fullfile('outdir', 'nade');

% load nade to sample from
num_nades = 60;
nades = cell(1, num_nades);
for i = 1:num_nades
    load(fullfile(outdir, 'ensemble', sprintf('mnist_allclass_500_logistic_%d.mat', i)), 'nade');
    nades{i} = nade;
end
ensemble = NadeEnsemble(nades, platform);
clear nades num_nades nade;

% generate samples
N = 100000;
x = zeros(ensemble.num_inputs, N, ensemble.arraytype);
L = zeros(1, N, ensemble.arraytype);
dLdx = zeros(ensemble.num_inputs, N, ensemble.arraytype);
bufsize = 500;

for i = 1:N/bufsize
    
    fprintf('%g %% \n', i*bufsize/N * 100);
    
    idx = (i-1)*bufsize+1 : i*bufsize;
    [x(:,idx), L(:,idx), dLdx(:,idx)] = ensemble.gen(bufsize);
end

% save samples
x = gather(x);
L = gather(L);
dLdx = gather(dLdx);
save(fullfile(outdir, 'samples_ensemble.mat'), 'x', 'L', 'dLdx');
