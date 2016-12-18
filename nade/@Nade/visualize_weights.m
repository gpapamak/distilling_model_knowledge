function visualize_weights(obj, layer, imsize, varargin)
% Displays the weights of a specified layer as images.
% INPUTS
%       layer      the layer whose weights to display
%       imsize     the image size
% -- optional name-value pairs
%       layout     number of rows and columns for each page
%       range      display range of pixel intensities
% 
% George Papamakarios, Jun 2015

p = inputParser;
p.addRequired('layer', @(t) t == 1 || t == 2);
p.addRequired('imsize', @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('layout', [1, 1], @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('range', [], @(t) isempty(t) || numel(t) == 2);
p.parse(layer, imsize, varargin{:});

switch layer
    case 1
        W = [obj.W; obj.c];
    case 2
        W = [obj.b obj.U];
end
W = W(obj.rev_order, :);

% create titles
titles = cell(1, size(W, 2));
for i = 1:size(W, 2)
    titles{i} = num2str(i);
end

% display weights
disp_imdata(W, imsize, 'titles', titles, 'layout', p.Results.layout, 'range', p.Results.range);
