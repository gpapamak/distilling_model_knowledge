function visualize_activations(obj, x, layers)
% Visualizes the activations of specified layers caused by given data
% minibatch.
% INPUTS
%       x        a minibatch of data
%       layers   list of layers to visualize activations of; defaults to 
%                the whole net except the input layer
% 
% George Papamakarios, Mar 2015

if nargin < 3
    layers = 1:obj.num_layers;
else
    check = @(t) isint(t) && all(1 <= t(:)) && all(t(:) <= obj.num_layers);
    assert(check(layers), 'Invalid layers.');
end

% make the arrangements for the platform
if obj.gpu
    createArray = @gpuArray;
else
    createArray = @(x) x;
end

obj.forwProp(createArray(x));

for l = unique(layers)
    
    figure;
    colormap(gray);
    imagesc(obj.layers{l+1}.x);
    colorbar;
    title(sprintf('Layer %d', l));

end

obj.clear();
