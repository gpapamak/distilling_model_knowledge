function [loss, iter, trace] = train_stream(obj, stream, varargin)
% Trains the network given a data steam by minimizing its loss function
% using stochastic gradient descent.
% INPUTS
%       stream       data stream which generates train data
% -- optional name-value pairs -- 
%       step         step size strategy (defaults to ADADELTA)
%       tol          termination tolerance for the difference between 
%                    function values
%       maxiter      maximum number of iterations
%       verbose      if true, print execution information
%       minibatch    minibatch size (defaults to 1)
%       wdecay       weight decay term coefficient (defaults to 0)
% OUTPUTS
%       loss         final value of the average loss
%       iter         total number of iterations
%       trace        progress of average loss after each iteration
% 
% George Papamakarios, Feb 2015

% assert that the network has a loss function
assert(~isempty(obj.loss_fun_hl), 'Network has no loss function and thus can''t be trained.');

% parse input
p = inputParser;
p.addRequired('stream', @(t) isa(t, 'DataStream'));
p.addParameter('step', AdaDelta(), @(t) isa(t, 'StepStrategy') || (isscalar(t) && isreal(t) && t > 0));
p.addParameter('tol', 0, @(t) isscalar(t) && isreal(t) && t >= 0);
p.addParameter('maxiter', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('verbose', true, @(t) isscalar(t) && islogical(t));
p.addParameter('minibatch', 1, @(t) isscalar(t) && isint(t) && t >= 1);
p.addParameter('wdecay', 0, @(t) isscalar(t) && isreal(t) && t >= 0);
p.parse(stream, varargin{:});

% tell the stream to produce derivatives if the loss function needs them
stream.genDerivs(obj.loss_fun_needs_derivs);

% set step
step = p.Results.step;
if ~isa(step, 'StepStrategy')
    step = ConstantStep(step);
end

% initialize
prev_loss = inf;
iter = 0;
N = p.Results.minibatch;
store_trace = nargout > 2;
if store_trace
    trace = [];
end

while true
    
    % get a mini-batch
    [x, y] = stream.gen(N);
    
    % compute loss and its derivatives
    [loss, dldp] = obj.eval_loss(x, y);
    
    % add a weight decay term
    loss = loss + p.Results.wdecay/2 * (obj.params' * obj.params);
    dldp = dldp + p.Results.wdecay * obj.params;
    diff = prev_loss - loss;

    % store and print info
    if store_trace
        trace = [trace, loss]; %#ok<AGROW>
    end
    if p.Results.verbose
        fprintf('Iteration %d, loss = %g, difference = %g \n', iter, loss, diff);
    end

    % check for convergence
    if abs(diff) <= p.Results.tol || iter >= p.Results.maxiter
        break;
    end

    % gradient update
    obj.setParamsFromVec(obj.params + step.next(dldp), true);
    iter = iter + 1;
    prev_loss = loss;
    
end

obj.clear();
