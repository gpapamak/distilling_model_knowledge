function disp_imdata(x, imsize, varargin)
% disp_imdata(x, imsize, varargin)
% Displays columns of a matrix as images.
% INPUTS
%       x        matrix containing vectorized images as columns
%       imsize   the true image size
% -- optional name-value pairs
%       titles   subtitles for each image
%       layout   number of rows and columns for each page
%       range    display range of pixel intensities
%
% George Papamakarios, Mar 2015

p = inputParser;
p.addRequired('x', @(t) isreal(t) && ismatrix(t));
p.addRequired('imsize', @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('titles', cell(1, size(x, 2)), @(t) iscellstr(t) && numel(t) == size(x, 2));
p.addParameter('layout', [1, 1], @(t) isint(t) && all(t(:) > 0) && numel(t) == 2);
p.addParameter('range', [], @(t) isempty(t) || numel(t) == 2);
p.parse(x, imsize, varargin{:});

titles = p.Results.titles;
layout = p.Results.layout;
range = p.Results.range;

num_ims = size(x, 2);
num_ims_per_page = prod(layout);
num_pages = ceil(num_ims / num_ims_per_page);
fig = figure;
i = 1;

while true
    
    % display next
    clf;
    for j = 1:num_ims_per_page
        idx = (i-1) * num_ims_per_page + j;
        if idx > num_ims, break; end
        im = reshape(x(:,idx), imsize);
        subplot(layout(1), layout(2), j);
        imshow(im, range);
        axis image;
        title(titles{idx});
    end
    suptitle(sprintf('Page %d', i));
    drawnow;
    
    % pause and wait for key press
    waitforbuttonpress;
    key = double(get(fig, 'CurrentCharacter'));
    switch key
        % left/up arrow
        case {28, 30}
            i = i - 1;
            if i < 1, i = num_pages; end
        % right/down arrow
        case {29, 31}
            i = i + 1;
            if i > num_pages, i = 1; end
        % escape
        case 27
            break;
        % spacebar
        case 32
            idx = input('Move to page: ');
            if 1 <= idx && idx <= num_pages, i = idx; end
    end
end

close(fig);
