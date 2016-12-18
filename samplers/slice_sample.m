function [samples, width] = slice_sample(logp, x, varargin)
% [samples, width] = slice_sample(logp, x, varargin)
% Slice sampling for multivariate continuous probability distributions.
% It cycles sampling from each conditional using univariate slice sampling.
% INPUT
%       logp      function handle of log distribution to sample from
%       x         sample to start from
%       ns        number of returned samples (optional, defaults to 1)
%       -- optional name-value pairs --
%       burnin    number of ignored samples in burn-in period (defaults to 0)
%       thin      keep every #thin samples (defaults to 1)
%       width     initial estimate of the width of the bracket (the algorithm adapts it during burn-in)
%       maxwidth  maximum slice width (prevents from endless looping in regions of very small probability)
% OUTPUT
%       samples   Dxnsamples matrix where columns are samples
%       width     final estimate of the width of the bracket
%
% George Papamakarios, Jan 2015

p = inputParser;
p.addRequired('logp', @(t) isa(t, 'function_handle'));
p.addRequired('x', @isreal);
p.addOptional('ns', 1, @(t) isscalar(t) && isint(t) && t > 0);
p.addParameter('burnin', 0, @(t) isscalar(t) && isint(t) && t >= 0);
p.addParameter('thin', 1, @(t) isscalar(t) && isint(t) && t > 0);
p.addParameter('width', 1, @(t) isscalar(t) && t > 0);
p.addParameter('maxwidth', inf, @(t) isscalar(t) && t > 0);
p.parse(logp, x, varargin{:});

sizx = size(x);
D = numel(x);
width = p.Results.width;

% throw away samples during burn-in
for s = 1:p.Results.burnin
    for d = randperm(D)
        cond_logp = @(t) logp(reshape([x(1:d-1), t, x(d+1:end)], sizx));
        [x(d), w] = sample_from_slice(cond_logp, x(d), width, p.Results.maxwidth);
        width = width + (w - width) / s; % NOTE that this is not correct sampling; but we throw burnin samples away anyway
    end
end

% start saving samples
samples = zeros(D, p.Results.ns);
for s = 1:p.Results.ns
    for i = 1:p.Results.thin
        for d = randperm(D)
            cond_logp = @(t) logp(reshape([x(1:d-1), t, x(d+1:end)], sizx));
            x(d) = sample_from_slice(cond_logp, x(d), width, p.Results.maxwidth);
        end
    end
    samples(:, s) = x(:);
end


function [x, w] = sample_from_slice(logp, cx, w, maxw)
% Samples uniformly from a slice by constructing a bracket.

% sample a slice uniformly
logu = logp(cx) - exprnd(1);

% position the bracket randomly around the current sample
lx = cx - w * rand;
ux = lx + w;

% find lower bracket end
while logp(lx) >= logu && cx - lx < maxw
    lx = lx - w;
end

% find upper bracket end
while logp(ux) >= logu && ux - cx < maxw
    ux = ux + w;
end

% sample uniformly from bracket
x = uniform_sample([lx, ux]);

% if outside slice, reject sample and shrink bracket
while logp(x) < logu
    if x < cx
        lx = x;
    else
        ux = x;
    end
    x = uniform_sample([lx, ux]);
end

% return the final width of the bracket
w = ux - lx;
