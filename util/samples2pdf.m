function [x, y] = samples2pdf(samples, nbins)
% [x, y] = samples2pdf(x, nbins)
% Given a set of 1D samples, calculates an approximation of the pdf.
% INPUT
%       samples   array of 1D samples
%       nbins     number of bins to use to approximate the pdf (optional)      
% OUTPUT
%       x         location of each bin
%       y         pdf value at each bin
%
% George Papamakarios, Jan 2015

nsamples = numel(samples);
if nargin < 2
    nbins = floor(sqrt(nsamples));
end

width = (max(samples(:)) - min(samples(:))) / nbins;
[y, x] = hist(samples, nbins);
y = y / (width * nsamples);
