classdef SoftmaxLayer < NeuralLayer
% Softmax neural network layer.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        dydx = []
        dxdz = []
    end

    methods (Access = public)
        
        % constructor
        function [obj] = SoftmaxLayer(num_units, arraytype)
            obj@NeuralLayer(num_units, arraytype);
        end
        
        % clears intermediate results
        function clear(obj)
            obj.dydx = [];
            obj.dxdz = [];
            clear@NeuralLayer(obj);
        end
        
        % changes platform
        function changePlatform(obj, platform)
            switch platform
                case 'cpu'
                    obj.dydx = gather(obj.dydx);
                    obj.dxdz = gather(obj.dxdz);
                case 'gpu'
                    obj.dydx = gpuArray(obj.dydx);
                    obj.dxdz = gpuArray(obj.dxdz);
                otherwise
                    error('Unknown platform.');
            end
            changePlatform@NeuralLayer(obj, platform);
        end
        
        % propagate forward
        function [x] = forwProp(obj, z)
            
            % subtract max(z) from all z to avoid overflow of exp(z)
            z = z - ones(obj.num_units, 1, obj.arraytype) * max(z, [], 1);
            x = exp(z);
            x = x ./ (ones(obj.num_units, 1, obj.arraytype) * sum(x, 1));
            obj.x = x;
        end
        
        % propagate backward
        function [dydz] = backProp(obj, dydx)
            
            obj.num_out = size(dydx, 2);
            N = size(dydx, 3);
            obj.dydx = dydx;
            
            xi = repmat(permute(obj.x, [1 3 2]), 1, obj.num_units, 1);
            xj = repmat(permute(obj.x, [3 1 2]), obj.num_units, 1, 1);
            obj.dxdz = obj.repout3(xj .* (repmat(eye(obj.num_units, obj.arraytype), 1, 1, N) - xi));
            
            buf = repmat(permute(dydx, [1 4 2 3]), 1, obj.num_units, 1, 1);
            dydz = permute(sum(buf .* obj.dxdz, 1), [2 3 4 1]);
        end
        
        % R{propagate forward}
        function [Rx] = RforwProp(obj, Rz)
            
            buf = repmat(permute(Rz, [1 4 2 3]), 1, obj.num_units, 1, 1);
            Rx = permute(sum(buf .* obj.dxdz, 1), [2 3 4 1]);
            
            obj.Rx = Rx;
        end
        
        % R{propagate backward}
        function [Rdydz] = RbackProp(obj, Rdydx)
            
            N = size(Rdydx, 3);
            
            xi = repmat(permute(obj.x, [1 4 3 2]), 1, obj.num_units, obj.num_out, 1);
            xj = repmat(permute(obj.x, [4 1 3 2]), obj.num_units, 1, obj.num_out, 1);
            Rxi = repmat(permute(obj.Rx, [1 4 2 3]), 1, obj.num_units, 1, 1);
            Rxj = repmat(permute(obj.Rx, [4 1 2 3]), obj.num_units, 1, 1, 1);
            Rdxdz = Rxj .* (repmat(eye(obj.num_units, obj.arraytype), 1, 1, obj.num_units, N) - xi) - xj .* Rxi;
            
            buf1 = repmat(permute(Rdydx, [4 1 2 3]), obj.num_units, 1, 1, 1) .* obj.dxdz;
            buf2 = repmat(permute(obj.dydx, [4 1 2 3]), obj.num_units, 1, 1, 1) .* Rdxdz;
            Rdydz = permute(sum(buf1, 2) + sum(buf2, 2), [1 3 4 2]);
        end
        
    end
end
