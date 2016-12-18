function [progress] = train_nade(obj, nade, varargin)
% Trains nade to mimic a given nade.
% INPUTS
%       nade      nade to mimic
% -- rest of inputs are as in train_stream()
% OUTPUTS
%       progress  the monitored training progress
% 
% George Papamakarios, Jun 2015

% simply set up a nade stream and pass it to train_stream
stream = NadeStream(nade);
[progress] = obj.train_stream(stream, varargin{:});
