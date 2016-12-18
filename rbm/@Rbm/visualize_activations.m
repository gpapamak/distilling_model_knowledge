function visualize_activations(obj, x)
% Visualizes the activations of the hidden units caused by a given data
% minibatch.
% INPUTS
%       x     a minibatch of data
% 
% George Papamakarios, Jun 2015

N = size(x, 2);
h = obj.sigm(obj.W' * x + obj.b * ones(1, N, obj.arraytype));

% hidden layer
figure;
colormap(gray);
imagesc(h);
colorbar;
title('Activations of hidden units');
