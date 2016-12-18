function [progress] = train_stream(obj, stream, varargin)
% Trains nade given a data steam by maximizing its average log probability
% using stochastic gradient descent.
% INPUTS
%       stream         data stream which generates train data
% -- optional name-value pairs -- 
%       loss           loss function to minimize
%       tol            termination tolerance for the difference between 
%                      average log probability values
%       maxiter        maximum number of iterations
%       minibatch      minibatch size (defaults to 1)
%       monitor_every  monitor progress every that many iterations
%       x_tst          if provided, a test set on which to monitor training
% OUTPUTS
%       progress       the monitored training progress
% 
% George Papamakarios, Jun 2015

% parse input
p = inputParser;
p.addRequired('stream', @(t) isa(t, 'DataStream'));
p.addParameter('loss', 'max_likelihood', @(t) (~ischar(t) && isreal(t) && numel(t) == 3 && all(t(:) >= 0)) || any(validatestring(t, {'max_likelihood', 'square_error', 'score_matching', 'variational'})));
p.addParameter('tol', 0, @(t) isscalar(t) && isreal(t) && t >= 0);
p.addParameter('maxiter', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('minibatch', 1, @(t) isscalar(t) && isint(t) && t >= 1);
p.addParameter('monitor_every', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('x_tst', zeros(obj.num_inputs, 0), @(t) ismatrix(t) && isreal(t) && size(t,1) == obj.num_inputs);
p.parse(stream, varargin{:});

% set loss
if ischar(p.Results.loss)
    switch p.Results.loss
        case 'max_likelihood'
            loss_fun = @max_likelihood;
        case 'square_error'
            loss_fun = @square_error;
        case 'score_matching'
            loss_fun = @score_matching;
        case 'variational'
            loss_fun = @variational;
    end
else
    loss_fun = @(obj, stream, N) all_losses(obj, stream, N, p.Results.loss);
end

% set step
step_W = AdaDelta();
step_c = AdaDelta();
step_U = AdaDelta();
step_b = AdaDelta();

% initialize
prev_L = -inf;
iter = 0;
N = p.Results.minibatch;
x_tst = obj.precision(p.Results.x_tst(obj.fwd_order, :));
monitor_tst = ~isempty(x_tst);
progress.trn = [];
if monitor_tst
    progress.tst = [];
end

while true
    
    [L, dLdW, dLdc, dLdU, dLdb] = loss_fun(obj, stream, N);
    diff = prev_L - L;

    % monitor training progress
    if mod(iter, p.Results.monitor_every) == 0
        
        progress.trn = [progress.trn, gather(L)];
        
        if monitor_tst
            obj.forwProp(x_tst);
            progress.tst = [progress.tst, gather(mean(obj.L))];
            fprintf('Iteration = %d, train loss = %g, test log prob = %g \n', iter, progress.trn(end), progress.tst(end));
        else
            fprintf('Iteration = %d, train loss = %g \n', iter, progress.trn(end));
        end
    end
    
    % check for convergence
    if abs(diff) <= p.Results.tol || iter >= p.Results.maxiter
        break;
    end

    % gradient update
    obj.W = obj.W + step_W.next(dLdW);
    obj.c = obj.c + step_c.next(dLdc);
    obj.U = obj.U + step_U.next(dLdU);
    obj.b = obj.b + step_b.next(dLdb);
    iter = iter + 1;
    prev_L = L;
    
end

obj.clear();


function [L, dLdW, dLdc, dLdU, dLdb] = max_likelihood(obj, stream, N)

% get a mini-batch
x = obj.precision(stream.gen(N));

% forward and back prop
obj.forwProp(x(obj.fwd_order, :));
obj.backProp();

% loss
L = -mean(obj.L);

% derivatives
dLdW = -mean(obj.dLdW, 3);
dLdc = -mean(obj.dLdc, 3);
dLdU = -mean(obj.dLdU, 3);
dLdb = -mean(obj.dLdb, 2);


function [L, dLdW, dLdc, dLdU, dLdb] = square_error(obj, stream, N)

% get a mini-batch
[x, T] = stream.gen(N);
x = obj.precision(x);
T = obj.precision(T);

% forward and back prop
obj.forwProp(x(obj.fwd_order, :));
obj.backProp();

% loss
c = 0;
err = obj.L - T + c;
L = mean(err .^ 2) / 2;

% derivatives
dLdb = mean(obj.dLdb .* (ones(obj.num_inputs, 1, obj.arraytype) * err), 2);
err = permute(err, [1 3 2]);
dLdW = mean(obj.dLdW .* repmat(err, obj.num_inputs-1, obj.num_hidden, 1), 3);
dLdc = mean(obj.dLdc .* repmat(err, 1, obj.num_hidden, 1), 3);
dLdU = mean(obj.dLdU .* repmat(err, obj.num_inputs, obj.num_hidden, 1), 3);


function [L, dLdW, dLdc, dLdU, dLdb] = score_matching(obj, stream, N)

% get a mini-batch
[x, ~, dTdx] = stream.gen(N);
x = obj.precision(x);
dTdx = obj.precision(dTdx);

% forward, backward and R prop
obj.forwProp(x(obj.fwd_order, :));
obj.backProp();
obj.backProp_inputs_reduced();
err = obj.dLdx - dTdx(obj.fwd_order, :);
obj.RbackProp(err);

% loss
L = mean(sum(err .^ 2, 1)) / 2;

% derivatives
dLdW = mean(obj.RdLdW, 3);
dLdc = mean(obj.RdLdc, 3);
dLdU = mean(obj.RdLdU, 3);
dLdb = mean(obj.RdLdb, 2);


function [L, dLdW, dLdc, dLdU, dLdb] = all_losses(obj, stream, N, lambda)

% get a mini-batch
[x, T, dTdx] = stream.gen(N);
x = obj.precision(x);
T = obj.precision(T);
dTdx = obj.precision(dTdx);

% forward and back prop
obj.forwProp(x(obj.fwd_order, :));
obj.backProp();

% max likelihood
L1 = -mean(obj.L);

dL1dW = -mean(obj.dLdW, 3);
dL1dc = -mean(obj.dLdc, 3);
dL1dU = -mean(obj.dLdU, 3);
dL1db = -mean(obj.dLdb, 2);

% square_error
err2 = obj.L - T;
L2 = mean(err2 .^ 2) / 2;

dL2db = mean(obj.dLdb .* (ones(obj.num_inputs, 1, obj.arraytype) * err2), 2);
err2 = permute(err2, [1 3 2]);
dL2dW = mean(obj.dLdW .* repmat(err2, obj.num_inputs-1, obj.num_hidden, 1), 3);
dL2dc = mean(obj.dLdc .* repmat(err2, 1, obj.num_hidden, 1), 3);
dL2dU = mean(obj.dLdU .* repmat(err2, obj.num_inputs, obj.num_hidden, 1), 3);

% score matching
obj.backProp_inputs_reduced();
err3 = obj.dLdx - dTdx(obj.fwd_order, :);
obj.RbackProp(err3);

L3 = mean(sum(err3 .^ 2, 1)) / 2;

dL3dW = mean(obj.RdLdW, 3);
dL3dc = mean(obj.RdLdc, 3);
dL3dU = mean(obj.RdLdU, 3);
dL3db = mean(obj.RdLdb, 2);

% combine all
lambda(3) = lambda(3) / size(x, 1);
L = lambda(1) * L1 + lambda(2) * L2 + lambda(3) * L3;
dLdW = lambda(1) * dL1dW + lambda(2) * dL2dW + lambda(3) * dL3dW;
dLdc = lambda(1) * dL1dc + lambda(2) * dL2dc + lambda(3) * dL3dc;
dLdU = lambda(1) * dL1dU + lambda(2) * dL2dU + lambda(3) * dL3dU;
dLdb = lambda(1) * dL1db + lambda(2) * dL2db + lambda(3) * dL3db;


function [L, dLdW, dLdc, dLdU, dLdb] = variational(obj, stream, N)

% get a mini-batch from nade
obj.forwProp_gen(N);
T = obj.precision(stream.rbm.eval(obj.x(obj.rev_order, :)));

% back prop
obj.backProp();

% loss
err = obj.L - T;
L = mean(err);

% derivatives
err = err + 1;
dLdb = mean(obj.dLdb .* (ones(obj.num_inputs, 1, obj.arraytype) * err), 2);
err = permute(err, [1 3 2]);
dLdW = mean(obj.dLdW .* repmat(err, obj.num_inputs-1, obj.num_hidden, 1), 3);
dLdc = mean(obj.dLdc .* repmat(err, 1, obj.num_hidden, 1), 3);
dLdU = mean(obj.dLdU .* repmat(err, obj.num_inputs, obj.num_hidden, 1), 3);
