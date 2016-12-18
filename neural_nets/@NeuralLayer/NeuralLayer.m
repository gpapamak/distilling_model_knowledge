classdef NeuralLayer < handle
% Abstract class for the non-linear part of a neural network layer.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = protected, GetAccess = public)
        
        arraytype = 'double'
        
        num_units = []
        num_out   = []
        
        x  = []
        Rx = []
    end
    
    methods (Access = protected)
        
        % repeats a [units x N] matrix to make it [units x out x N]
        function [y] = repout2(obj, x)
            y = repmat(permute(x, [1 3 2]), 1, obj.num_out, 1);
        end
        
        % repeats a [units x units x N] matrix to make it [units x units x out x N]
        function [y] = repout3(obj, x)
            y = repmat(permute(x, [1 2 4 3]), 1, 1, obj.num_out, 1);
        end
    end

    methods (Access = public)
        
        % constructor
        function [obj] = NeuralLayer(num_units, arraytype)
            obj.num_units = num_units;
            obj.arraytype = arraytype;
        end
        
        % clears intermediate results
        function clear(obj)
            obj.x  = [];
            obj.Rx = [];
        end
        
        % changes platform
        function changePlatform(obj, platform)
            switch platform
                case 'cpu'
                    obj.arraytype = 'double';
                    obj.x  = gather(obj.x);
                    obj.Rx = gather(obj.Rx);
                case 'gpu'
                    obj.arraytype = 'gpuArray';
                    obj.x  = gpuArray(obj.x);
                    obj.Rx = gpuArray(obj.Rx);
                otherwise
                    error('Unknown platform.');
            end
        end
    end

    methods (Abstract, Access = public)
        
        % propagate forward
        [x] = forwProp(obj, z)
        
        % propagate backward
        [dydz] = backProp(obj, dydx)
        
        % R{propagate forward}
        [Rx] = RforwProp(obj, Rz)
        
        % R{propagate backward}
        [Rdydz] = RbackProp(obj, Rdydx)
    end
end
