function param_hist(obj, layers)
% Displays a histogram of weights and biases for specified layers.
% INPUTS
%       layers   list of layers to show histograms for; defaults to the
%                whole net
% 
% George Papamakarios, Mar 2015

if nargin < 2
    layers = 1:obj.num_layers;
else
    check = @(t) isint(t) && all(1 <= t(:)) && all(t(:) <= obj.num_layers);
    assert(check(layers), 'Invalid layers.');
end

for l = unique(layers)
    
    figure;
    suptitle(sprintf('Layer %d', l));
    
    subplot(1,2,1);
    nbins = floor(sqrt(numel(obj.weights{l})));
    hist(obj.weights{l}(:), nbins);
    title('weights');
    
    subplot(1,2,2);
    nbins = floor(sqrt(numel(obj.biases{l})));
    hist(obj.biases{l}(:), nbins);
    title('biases');

end
