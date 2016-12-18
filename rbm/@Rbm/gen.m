function [x, L, dLdx] = gen(obj, N, thin)
% Generates samples from rbm, using block gibbs sampling. If requested,
% also outputs the unnormalized log probability and its derivatives at the
% locations of the samples.
% INPUT
%     N       number of samples to generate
%     thin    how much to thin (optional; defaults to 1, ie no thinning)
% OUTPUT
%     x       columns are samples
%     L       log p(x) (unnormalized)
%     dLdx    columns are derivatives of log p(x)
% 
% George Papamakarios, Jun 2015

if nargin < 3
    thin = 1;
end

% check input
check = @(t) isscalar(t) && isint(t) && t > 0;
assert(check(N), 'Number of samples must be a positive integer.');
assert(check(thin), 'Thinning amount must be a positive integer.');
assert(mod(N, obj.num_chains) == 0, 'Number of samples must be a multiple of the number of chains,');

x = zeros(obj.num_inputs, N, obj.arraytype);

% -- sample

v = obj.gibbs_state;
aa = obj.a * ones(1, obj.num_chains, obj.arraytype);
bb = obj.b * ones(1, obj.num_chains, obj.arraytype);

for n = 1:obj.num_chains:N
    for t = 1:thin
    
        h = obj.sigm(obj.W' * v + bb);
        h = double(rand(obj.num_hidden, obj.num_chains, obj.arraytype) < h);

        v = obj.sigm(obj.W * h + aa);
        v = double(rand(obj.num_inputs, obj.num_chains, obj.arraytype) < v);
    end
    
    x(:, n : n+obj.num_chains-1) = v;
end

obj.gibbs_state = v;

% -- evaluate

if nargout == 2
    
    L = obj.eval(x);
    
elseif nargout == 3 
    
    [L, dLdx] = obj.eval(x);

end
