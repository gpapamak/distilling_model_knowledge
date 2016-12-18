function [y] = log_mog_pdf(x, a, varargin)
% [y] = log_mog_pdf(x, a, varargin)
% Log pdf of a mixture of gaussians. Accepts a set of N input locations
% and a set of K parameters, and it returns the log pdf evaluated at all 
% KxN combinations.
% INPUT
%       x         DxN matrix, each column is an input location
%       a         MxK matrix, each column is a list of mixture coefficients
%       varargin  either w or (m, S) where:
%                 w   (D+D^2)xMxK array, each slice is a vectorised list of
%                     parameters (m, S)
%                 m   DxMxK array, each slice contains component means
%                 S   DxDxMxK array, each []x[]x[]xk subarray contains 
%                     component covariances
% OUTPUT
%       y         KxN matrix of log pdf values
%
% George Papamakarios, Jan 2015

D = size(x, 1);
[M, K] = size(a);

% parse inputs
switch nargin
    case 3
        w = varargin{1};
        assert(size(w, 1) == D + D^2 && size(w, 2) == M && size(w, 3) == K, 'Sizes don''t match.');
        
    case 4
        m = varargin{1};
        S = varargin{2};
        if D == 1
            S = reshape(S, 1, 1, M, K);
        end
        assert(size(m, 1) == D && size(S, 1) == D && size(S, 2) == D, 'Sizes don''t match.');
        assert(size(S, 3) == M && size(S, 4) == K, 'Sizes don''t match.');
        w = [m; reshape(S, D^2, M, K)];
        
    otherwise
        error('Function takes either 3 or 4 arguments.');
end

% calculate log pdf
y = log_mixture_pdf(x, @log_gauss_pdf, a, w);  
