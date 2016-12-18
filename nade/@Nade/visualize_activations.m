function visualize_activations(obj, x)
% Visualizes the activations caused by a given data minibatch.
% INPUTS
%       x        a minibatch of data
% 
% George Papamakarios, Jun 2015

N = size(x, 2);
obj.forwProp(obj.precision(x(obj.fwd_order, :)));

% hidden layer
figure;
colormap(gray);
imagesc(reshape(obj.h(obj.rev_order, :), [], N));
colorbar;
title('Hidden layer');

% output layer
figure;
colormap(gray);
imagesc(obj.y(obj.rev_order, :));
colorbar;
title('Output layer');

obj.clear();
