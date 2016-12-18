function [x, accept] = metropolis_gauss(logp, step, x1, varargin)
% [x, accept] = metropolis_gauss(logp, step, x1, varargin)
% Metropolis algorithm with a spherical gaussian as proposal distribution.
% Samples from a (not necessarily normalized) multivariate continuous 
% probability distribution.
% INPUT
%       logp      function handle of log distribution to sample from
%       step      standard deviation of gaussian proposal
%       x1        sample to start from
%       nsamples  number of returned samples (optional, defaults to 1)
%       -- optional name-value pairs --
%       burnin    number of ignored samples in burn-in period (defaults to 0)
%       thin      keep every #thin samples (defaults to 1)
% OUTPUT
%       x         Dxnsamples matrix where columns are samples
%       accept    proportion of accepted proposals
%
% George Papamakarios, Jan 2015

p = inputParser;
p.addRequired('logp', @(t) isa(t, 'function_handle'));
p.addRequired('step', @(t) isscalar(t) && t > 0);
p.addRequired('x1', @isreal);
p.addOptional('nsamples', 1, @(t) isscalar(t) && isint(t) && t > 0);
p.addParameter('burnin', 0, @(t) isscalar(t) && isint(t) && t >= 0);
p.addParameter('thin', 1, @(t) isscalar(t) && isint(t) && t > 0);
p.parse(logp, step, x1, varargin{:});

sizx = size(x1);
x_prev = x1;
p_prev = logp(x1);

% burn-in period, samples ignored
for s = 1:p.Results.burnin
    
    % propose a new sample
    x_prop = x_prev + step * randn(sizx);
    p_prop = logp(x_prop);
    
    if log(rand) < p_prop - p_prev
        % accept proposal
        x_prev = x_prop;
        p_prev = p_prop;
    end
end

x = zeros(numel(x1), p.Results.nsamples);
accept = 0;

% actual sampling
for s = 1:p.Results.nsamples
    
    for t = 1:p.Results.thin
        % propose a new sample
        x_prop = x_prev + step * randn(sizx);
        p_prop = logp(x_prop);

        if log(rand) < p_prop - p_prev
            % accept proposal
            x_prev = x_prop;
            p_prev = p_prop;
            accept = accept + 1;
        end
    end
    
    x(:,s) = x_prev(:);
end

accept = accept / p.Results.nsamples;
