function [samples, idx] = data_sample(data, replacement, num_samples)
% [samples] = data_sample(data, replacement, num_samples)
% Samples from a data set with or without replacement.
% INPUTS
%       data           a data matrix with datapoints as columns
%       replacement    true to sample with replacement, false otherwise
%       num_samples    number of samples to draw
% OUTPUTS
%       samples        matrix with samples as columns
%       idx            indices of the sampled points
%
% NOTE: if sampling without replacement, num_samples must not be larger
% that the number of samples; otherwise an exception is thrown.
% 
% George Papamakarios, Apr 2015

num_data = size(data, 2);

if replacement
    idx = randi(num_data, [1 num_samples]);
else
    idx = randperm(num_data, num_samples);
end

samples = data(:, idx);
