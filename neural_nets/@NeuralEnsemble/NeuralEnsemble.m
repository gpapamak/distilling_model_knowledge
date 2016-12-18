classdef NeuralEnsemble < handle
% An ensemble of neural nets.
%
% George Papamakarios, Apr 2015
    
    properties (SetAccess = private, GetAccess = public)
        
        nets = {}
        type = 'mean'
        platform = 'cpu'
        
        num_nets    = 0
        num_inputs  = 0
        num_outputs = 0
        num_params  = 0

    end
    
    methods (Access = public)
        
        % constructs an ensemble given a set of nets
        function [obj] = NeuralEnsemble(nets, type, platform)
            
            if nargin < 3
                platform = 'cpu';
            end
            if nargin < 2
                type = 'mean';
            end
            assert(any(validatestring(type, {'mean', 'logmeanexp'})), 'Unsupported ensemble type.');
            assert(any(validatestring(platform, {'cpu', 'gpu'})), 'Unknown platform.');
            obj.type = type;
            obj.platform = platform;
            
            assert(iscell(nets), 'Input must be a cell array of neural nets.');
            assert(~isempty(nets), 'Input can''t be empty.');
            
            obj.num_nets = numel(nets);
            
            assert(isa(nets{1}, 'NeuralNet'), 'Input must be a cell array of neural nets.');
            obj.num_inputs  = nets{1}.num_inputs;
            obj.num_outputs = nets{1}.num_outputs;
            obj.num_params  = nets{1}.num_params;
            nets{1}.changePlatform(platform);
            
            for i = 2:obj.num_nets
                assert(isa(nets{i}, 'NeuralNet'), 'Input must be a cell array of neural nets.');
                assert(nets{i}.num_inputs  == obj.num_inputs,  'All nets must have the same number of inputs.');
                assert(nets{i}.num_outputs == obj.num_outputs, 'All nets must have the same number of outputs.');
                obj.num_params = obj.num_params + nets{i}.num_params;
                nets{i}.changePlatform(platform);
            end
            
            obj.nets = nets(:);
        end
        
        % evaluates ensemble for a given input
        function [y, dydx] = eval(obj, x)
           
            switch obj.type
                
                case 'mean'
            
                    y = 0;

                    if nargout > 1

                        dydx = 0;
                        for i = 1:obj.num_nets
                            [yi, dyidx] = obj.nets{i}.eval(x);
                            y = y + yi;
                            dydx = dydx + dyidx;
                        end
                        dydx = dydx / obj.num_nets;

                    else
                        for i = 1:obj.num_nets
                            y = y + obj.nets{i}.eval(x);
                        end
                    end

                    y = y / obj.num_nets;
                    
                case 'logmeanexp'
                
                    y = 0;

                    if nargout > 1

                        dydx = 0;
                        
                        for i = 1:obj.num_nets
                            [yi, dyidx] = obj.nets{i}.eval(x);
                            yi = exp(yi);
                            y = y + yi;
                            dydx = dydx + repmat(permute(yi, [3 1 2]), obj.num_inputs, 1, 1) .* dyidx;
                        end
                        dydx = dydx ./ repmat(permute(y, [3 1 2]), obj.num_inputs, 1, 1);

                    else
                        for i = 1:obj.num_nets
                            y = y + exp(obj.nets{i}.eval(x));
                        end
                    end

                    y = log(y) - log(obj.num_nets);

            end
        end
        
        % changes the platform to cpu or gpu
        function changePlatform(obj, platform)
            
            assert(any(validatestring(platform, {'cpu', 'gpu'})), 'Unknown platform.');
            obj.platform = platform;
            
            for i = 1:obj.num_nets
                obj.nets{i}.changePlatform(platform);
            end
            
        end
        
    end
end
