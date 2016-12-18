function [progress] = train(obj, x, varargin)
% Trains nade given a set of samples by maximizing average log probability
% using stochastic gradient descent.
% INPUTS
%       x            input samples
% -- rest of inputs are as in train_stream()
% OUTPUTS
%       progress     the monitored training progress
% 
% George Papamakarios, Jun 2015

% check input
check = @(t) isreal(t) && ismatrix(t) && size(t,1) == obj.num_inputs;
assert(check(x), 'Invalid input.');

% set up a data stream and pass it to train_stream
stream = DataSubSampler(obj.precision(x));
progress = obj.train_stream(stream, varargin{:});
