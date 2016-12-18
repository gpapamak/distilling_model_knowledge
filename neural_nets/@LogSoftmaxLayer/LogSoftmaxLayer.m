classdef LogSoftmaxLayer < NeuralLayer
% Logarithmic softmax neural network layer.
%
% George Papamakarios, Apr 2015

    properties (SetAccess = private, GetAccess = public)
                
        ex   = []
        dydx = []
    end

    methods (Access = public)
        
        % constructor
        function [obj] = LogSoftmaxLayer(num_units, arraytype)
            obj@NeuralLayer(num_units, arraytype);
        end
        
        % clears intermediate results
        function clear(obj)
            obj.ex   = [];
            obj.dydx = [];
            clear@NeuralLayer(obj);
        end
        
        % changes platform
        function changePlatform(obj, platform)
            switch platform
                case 'cpu'
                    obj.ex   = gather(obj.ex);
                    obj.dydx = gather(obj.dydx);
                case 'gpu'
                    obj.ex   = gpuArray(obj.ex);
                    obj.dydx = gpuArray(obj.dydx);
                otherwise
                    error('Unknown platform.');
            end
            changePlatform@NeuralLayer(obj, platform);
        end
        
        % propagate forward
        function [x] = forwProp(obj, z)
            
            % subtract max(z) from all z to avoid overflow of exp(z)
            z = z - ones(obj.num_units, 1, obj.arraytype) * max(z, [], 1);
            x = z - ones(obj.num_units, 1, obj.arraytype) * log(sum(exp(z), 1));
            obj.x = x;
        end
        
        % propagate backward
        function [dydz] = backProp(obj, dydx)
            
            obj.num_out = size(dydx, 2);
            obj.dydx = dydx;
            obj.ex = obj.repout2(exp(obj.x));
            
            dydz = dydx - obj.ex .* repmat(sum(dydx, 1), obj.num_units, 1, 1);
        end
        
        % R{propagate forward}
        function [Rx] = RforwProp(obj, Rz)
            
            Rx = Rz - repmat(sum(obj.ex .* Rz, 1), obj.num_units, 1, 1);
            obj.Rx = Rx;
        end
        
        % R{propagate backward}
        function [Rdydz] = RbackProp(obj, Rdydx)
            
            Rdydz = Rdydx - obj.ex .* (repmat(sum(Rdydx, 1), obj.num_units, 1, 1) + obj.Rx .* repmat(sum(obj.dydx, 1), obj.num_units, 1, 1));
        end
        
    end
end
