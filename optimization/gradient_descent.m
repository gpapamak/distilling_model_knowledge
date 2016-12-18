function [x, info] = gradient_descent(x, f, step, varargin)
% [x, info] = gradient_descent(x, f, step, varargin)
% Gradient descent for function minimization.
% INPUTS
%       x         starting point
%       f         handle that returns the function value and the gradient
%       step      step size (allowed to be negative to get gradient ascent)
%       -- optional name-value pairs -- 
%       tol       termination tolerance for the difference between function values
%       maxiter   maximum number of iterations
%       verbose   if true, print execution information
% OUTPUTS
%       x         local minimizer
%       info      if asked for, a struct with execution information
%                 note that storing all the info might be constly
%
% George Papamakarios, Jan 2015

p = inputParser;
p.addRequired('x', @isreal);
p.addRequired('f', @(t) isa(t, 'function_handle'));
p.addRequired('step', @(t) isscalar(t) && isreal(t));
p.addParameter('tol', 1.0e-8, @(t) isscalar(t) && isreal(t) && t >= 0);
p.addParameter('maxiter', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('verbose', false, @(t) isscalar(t) && islogical(t));
p.parse(x, f, step, varargin{:});

% initialize
iter = 0;
diff = inf;
[fx, dfx] = f(x);
store_info = nargout > 1;
if store_info
    info.iter = iter;
    info.diff = diff;
    info.x = x(:);
    info.f = fx;
end

% iterate
while abs(diff) > p.Results.tol && iter < p.Results.maxiter
    
    % make an update
    x = x - step * dfx;
    [fx_new, dfx] = f(x);
    diff = fx - fx_new;
    fx = fx_new;
    iter = iter + 1;
    
    % print info
    if p.Results.verbose
        fprintf('Iteration %d, function = %g, difference = %g \n', iter, fx, diff);
    end
    
    % store info
    if store_info
        info.iter = [info.iter, iter];
        info.diff = [info.diff, diff];
        info.x = [info.x, x(:)];
        info.f = [info.f, fx];
    end
    
end
