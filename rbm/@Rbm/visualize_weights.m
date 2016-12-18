function visualize_weights(obj, imsize, varargin)
% Displays the weights as images.
% INPUTS
%       imsize     the image size
% -- optional name-value pairs
%       layout     number of rows and columns for each page
%       range      display range of pixel intensities
% 
% George Papamakarios, Jun 2015

p = inputParser;
p.addRequired('imsize', @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('layout', [1, 1], @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('range', [], @(t) isempty(t) || numel(t) == 2);
p.parse(imsize, varargin{:});

% create titles
titles = cell(1, obj.num_hidden + 1);
titles{1} = 'bias';
for i = 2:numel(titles)
    titles{i} = num2str(i-1);
end

% display weights
disp_imdata([obj.a, obj.W], imsize, 'titles', titles, 'layout', p.Results.layout, 'range', p.Results.range);
