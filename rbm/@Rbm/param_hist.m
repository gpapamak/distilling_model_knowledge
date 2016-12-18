function param_hist(obj)
% Displays a histogram of all parameters of the rbm.
% 
% George Papamakarios, Jun 2015

% weights
figure;
nbins = floor(sqrt(numel(obj.W)));
hist(obj.W(:), nbins);
title('Weights');

% input biases
figure;
nbins = floor(sqrt(numel(obj.a)));
hist(obj.a(:), nbins);
title('Input biases');

% hidden biases
figure;
nbins = floor(sqrt(numel(obj.b)));
hist(obj.b(:), nbins);
title('Hidden biases');
