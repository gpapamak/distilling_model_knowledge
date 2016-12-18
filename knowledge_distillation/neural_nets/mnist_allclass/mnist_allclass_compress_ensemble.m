% Compress an ensemble of neural nets trained on mnist to a single small
% neural net.
%
% George Papamakarios, May 2015

clear;
close all;
outdir = fullfile('outdir', 'neural_nets', 'mnist_allclass');
platform = 'gpu';

% select type of compression
final_layer = 'softmax'; ensemble_type = 'mean'; loss_function = 'dot_product'; outfile = 'compress_ensemble_cross_entropy_mnist_60k.mat';
%final_layer = 'logsoftmax'; ensemble_type = 'logmeanexp'; loss_function = 'deriv_square_error'; outfile = 'compress_ensemble_deriv_square_error_mnist_60k.mat';

% make the arrangements for the platform
switch platform
    case 'cpu'
        createArray = @(x) x;
        arraytype = 'double';
    case 'gpu'
        createArray = @gpuArray;
        arraytype = 'gpuArray';
    otherwise
        error('Uknown platform.');
end

%% load ensemble
num_nets = 30;
nets = cell(1, num_nets);

for i = 1:num_nets
    
    load(fullfile(outdir, 'ensemble', sprintf('net_%d.mat', i)), 'net');
    net.changeLayerType('final', final_layer);
    nets{i} = net;
end

ensemble = NeuralEnsemble(nets, ensemble_type, platform);
clear num_nets nets net i;

%% load mnist
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'), 'x_trn', 'x_tst', 'y_trn', 'y_tst');
x_trn = createArray(x_trn(:, 1:60000));
x_tst = createArray(x_tst);
y_trn = createArray(y_trn(:, 1:60000));
y_tst = createArray(y_tst);
num_trn = size(x_trn, 2);
num_tst = size(x_tst, 2);
[~, y_tst_lb] = max(y_tst);
y_tst_lb = y_tst_lb - 1;

%% load nade
load(fullfile('outdir', 'nade', 'mnist_allclass_500_logistic_60k.mat'), 'nade');
nade.changePlatform(platform);

%% compress ensemble
net = NeuralNet(ensemble.num_inputs, platform);
net.addLayer(50, 'type', 'relu');
net.addLayer(30, 'type', 'relu');
net.addLayer(10, 'type', 'logsoftmax');
net.setLossFunction(loss_function);

% train the net
passes = 40;
batch_size = 20;
num_epochs = passes * floor(num_trn / batch_size);
check_every = 100;
all_acc = [];
all_logprob = [];
%stream = NetworkSampler(ensemble, @(N) randn(ensemble.num_inputs, N, arraytype));
%stream = NetworkSampler(ensemble, nade);
stream = NetworkSampler(ensemble, x_trn);
stream.genDerivs(net.loss_fun_needs_derivs);
step = AdaDelta();

for i = 1:floor(num_epochs / check_every)
    
    [x_data, y_data] = stream.gen(batch_size * check_every);
    net.train_stream(DataSubSampler(x_data, y_data), 'step', step, 'minibatch', batch_size, 'maxiter', check_every, 'verbose', false);
    
    % performance on the test set
    y = net.eval(x_tst);
    [~, y_prd] = max(y);
    y_prd = y_prd - 1;
    acc = y_prd == y_tst_lb;
    acc_avg = mean(acc);
    acc_std = std(acc) / sqrt(num_tst);
    logprob = y(logical(y_tst));
    logprob_avg = mean(logprob);
    logprob_std = std(logprob) / sqrt(num_tst);
    fprintf('Iteration = %d, accuracy = %g%% +- %g%%, log-probability = %g +- %g \n', i, acc_avg*100, 2*acc_std*100, logprob_avg, 2*logprob_std);
    all_acc = [all_acc, acc_avg]; %#ok<AGROW>
    all_logprob = [all_logprob, logprob_avg]; %#ok<AGROW>
end

%% save the net and training progress
net.changePlatform('cpu');
all_acc = gather(all_acc);
all_logprob = gather(all_logprob);
save(fullfile(outdir, outfile), 'net', 'all_acc', 'all_logprob', 'passes', 'batch_size', 'check_every');
