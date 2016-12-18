function visualize_weights(obj, layer, imsize, varargin)
% Displays the weights of a specified layer as images.
% INPUTS
%       layer      the layer whose weights to display
%       imsize     the image size
% -- optional name-value pairs
%       layout     number of rows and columns for each page
%       range      display range of pixel intensities
% 
% George Papamakarios, Mar 2015

p = inputParser;
p.addRequired('layer', @(t) isscalar(t) && isint(t) && 1 <= t && t <= obj.num_layers);
p.addRequired('imsize', @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('layout', [1, 1], @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('range', [min(obj.weights{layer}(:)), max(obj.weights{layer}(:))], @(t) isempty(t) || numel(t) == 2);
p.parse(layer, imsize, varargin{:});

% create titles
titles = cell(1, obj.num_units(layer+1));
for i = 1:obj.num_units(layer+1)
    titles{i} = num2str(i);
end

% display weights
disp_imdata(obj.weights{layer}', imsize, 'titles', titles, 'layout', p.Results.layout, 'range', p.Results.range);
