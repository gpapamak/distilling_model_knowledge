function [L, dLdx] = eval(obj, x)
% Evaluates the unnormalized log likelihood of the rbm and its derivatives
% at specified input locations.
% INPUT
%     x     columns are input locations
% OUTPUT
%     L     log p(x) (unnormalized) 
%     dLdx  derivatives of log p(x) wrt x
% 
% George Papamakarios, Jun 2015

N = size(x, 2);

h = obj.W' * x + obj.b * ones(1, N, obj.arraytype);

% if h > 0 then log(1+exp(h)) = h + log(1+exp(-h)) to avoid overflow of exp
idx = h > 0;
hp = zeros(obj.num_hidden, N, obj.arraytype);
hm = h;
hp(idx) =  h(idx);
hm(idx) = -h(idx);
log1pexph = hp + log1p(exp(hm));

L = obj.a' * x + sum(log1pexph, 1) - obj.logZ;

if nargout > 1
    dLdx = obj.W * obj.sigm(h) + obj.a * ones(1, N, obj.arraytype);
end
