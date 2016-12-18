function [y] = log_gauss_pdf(x, varargin)
% [y] = log_gauss_pdf(x, varargin)
% Log pdf of a multivariate gaussian. Accepts a set of N input locations
% and a set of K parameters, and it returns the log pdf evaluated at all 
% KxN combinations.
% INPUT
%       x         DxN matrix, each column is an input location
%       varargin  either w or (m, S) where:
%                 w   (D+D^2)xK matrix, each column is a vectorised list of
%                     parameters (m, S)
%                 m   DxK matrix, each column is a mean vector
%                 S   DxDxK array, each slice is a covariance matrix
% OUTPUT
%       y         KxN matrix of log pdf values
%
% George Papamakarios, Jan 2015

% treat the 1d case separately, for speed
if size(x, 1) == 1
    y = log_gauss_pdf_1d(x, varargin{:});
else
    y = log_gauss_pdf_nd(x, varargin{:});
end


function [y] = log_gauss_pdf_1d(x, varargin)

N = length(x);

% parse inputs
switch nargin
    case 2
        w = varargin{1};
        assert(size(w, 1) == 2, 'Sizes don''t match.');
        K = size(w, 2);
        m = w(1, :)';
        S = w(2, :)';
        
    case 3
        m = varargin{1}; m = m(:);
        S = varargin{2}; S = S(:);
        K = length(m);
        assert(length(S) == K, 'Sizes don''t match.');
        
    otherwise
        error('Function takes either 2 or 3 arguments.');
end

% calculate log pdf
xm = ones(K, 1) * x - m * ones(1, N);
denom = log((2*pi) * S);
y = (xm.^2) ./ (S * ones(1, N)) + denom * ones(1, N);
y = -y / 2;


function [y] = log_gauss_pdf_nd(x, varargin)

[D, N] = size(x);

% parse inputs
switch nargin
    case 2
        w = varargin{1};
        assert(size(w, 1) == D + D^2, 'Sizes don''t match.');
        K = size(w, 2);
        m = w(1:D, :);
        S = reshape(w(D+1:end, :), D, D, K);
        
    case 3
        m = varargin{1};
        S = varargin{2};
        K = size(m, 2);
        assert(size(m, 1) == D && size(S, 1) == D && size(S, 2) == D, 'Sizes don''t match.');
        assert(size(S, 3) == K, 'Sizes don''t match.');
        
    otherwise
        error('Function takes either 2 or 3 arguments.');
end

% calculate log pdf
% note: replace log(det()) with logdet('pd') in case of over/underflow
% it is slower but numerically more stable
y = zeros(K, N);
for k = 1:K
    xm = x - m(:,k) * ones(1, N);
    y(k,:) = sum(xm .* (S(:,:,k) \ xm), 1) + log(det((2*pi) * S(:,:,k)));
end
y = -y / 2;
