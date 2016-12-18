function [loss, iter, trace] = train_net(obj, net, inp, varargin)
% Trains the network to mimic a given network.
% INPUTS
%       net          network to mimic
%       inp          either a sampler of input locations or a dataset
% -- rest of inputs are as in train_stream()
% OUTPUTS
%       loss         final value of the average loss
%       iter         total number of iterations
%       trace        progress of average loss after each iteration
% 
% George Papamakarios, Apr 2015

% simply set up a stream and pass it to train_stream
stream = NetworkSampler(net, inp);
[loss, iter, trace] = obj.train_stream(stream, varargin{:});
