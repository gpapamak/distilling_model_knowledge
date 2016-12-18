% Trains a neural net on the mnist dataset to be used as a component of the
% neural ensemble. Includes the option to bag the dataset. Saves the net.
%
% George Papamakarios, May 2015

% set an id for the net and choose whether to bag the dataset
id = 30;
bagging = true;

rng('shuffle');
seed = rng; %#ok<NASGU>

% load mnist data
load(fullfile('..', 'data', 'mnist', 'all_digits.mat'));
[num_inputs, num_trn] = size(x_trn);

% if bagging is on, resample mnist with replacement
if bagging
    [x_trn, idx] = data_sample(x_trn, true, num_trn);
    y_trn = y_trn(:, idx);
end

% create the net
net = NeuralNet(num_inputs);
net.addLayer(500, 'type', 'relu');
net.addLayer(300, 'type', 'relu');
net.addLayer( 10, 'type', 'logsoftmax');
net.setLossFunction('dot_product');

% train the net
passes = 20;
batch_size = 20;
num_epochs = passes * floor(num_trn / batch_size);
net.train(x_trn, y_trn, 'minibatch', batch_size, 'maxiter', num_epochs);

% save the net and quit
outdir = fullfile('outdir', 'neural_nets', 'mnist_allclass');
save(fullfile(outdir, 'ensemble', sprintf('net_%d.mat', id)), 'net', 'batch_size', 'passes', 'bagging', 'seed');
quit;
