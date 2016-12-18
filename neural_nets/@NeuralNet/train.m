function [loss, iter, trace] = train(obj, x, y, varargin)
% Trains the network given a set of examples by minimizing a specified loss
% function using stochastic gradient descent.
% INPUTS
%       x            input train data
%       y            target train data
% -- rest of inputs are as in train_stream()
% OUTPUTS
%       loss         final value of the average loss
%       iter         total number of iterations
%       trace        progress of average loss after each iteration
% 
% George Papamakarios, Feb 2015

N = size(x, 2);

% parse input
p = inputParser;
p.addRequired('x', @(t) isreal(t) && ismatrix(t) && size(t,1) == obj.num_inputs);
p.addRequired('y', @(t) isreal(t) && ismatrix(t) && size(t,2) == N);
p.parse(x, y);

% set up a data stream and pass it to train_stream
stream = DataSubSampler(x, y);
[loss, iter, trace] = obj.train_stream(stream, varargin{:});
