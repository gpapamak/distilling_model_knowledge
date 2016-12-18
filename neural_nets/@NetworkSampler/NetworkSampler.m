classdef NetworkSampler < DataStream
% Given a sampler S and a neural network (or neural ensemble) N, samples
% inputs with S and evaluates N on them. As a result, it generates batches
% of input-output pairs.
%
% George Papamakarios, Feb 2015

    properties (SetAccess = private, GetAccess = public)
        
        net = []
        input_sampler = []
        
    end
    
    methods (Access = public)
        
        % constructor
        function [obj] = NetworkSampler(net, input, varargin)
        
            p = inputParser;
            p.addRequired('net', @(t) isa(t, 'NeuralNet') || isa(t, 'NeuralEnsemble'));
            p.addRequired('input', @(t) isa(t, 'function_handle') || isa(t, 'Nade') || isreal(t) && ismatrix(t) && size(t,1) == net.num_inputs);
            p.addParameter('gen_derivs', false, @(t) isscalar(t) && islogical(t));
            p.parse(net, input, varargin{:});
            
            if isa(input, 'function_handle')
                obj.input_sampler = input;
            elseif isa(input, 'Nade')
                obj.input_sampler = @input.gen;
            else
                stream = DataSubSampler(input);
                obj.input_sampler = @stream.gen;
            end
            
            obj.net = net;
            obj.gen_derivs = p.Results.gen_derivs;
        end
        
        % generates a new data batch
        [x, y] = gen(obj, N)
        
    end
end
