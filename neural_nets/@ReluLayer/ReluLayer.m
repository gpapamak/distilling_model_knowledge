classdef ReluLayer < NeuralLayer
% Rectified linear unit (relu) neural network layer.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        dxdz = []
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ReluLayer(num_units, arraytype)
            obj@NeuralLayer(num_units, arraytype);
        end
        
        % clears intermediate results
        function clear(obj)
            obj.dxdz = [];
            clear@NeuralLayer(obj);
        end
        
        % changes platform
        function changePlatform(obj, platform)
            switch platform
                case 'cpu'
                    obj.dxdz = gather(obj.dxdz);
                case 'gpu'
                    obj.dxdz = gpuArray(obj.dxdz);
                otherwise
                    error('Unknown platform.');
            end
            changePlatform@NeuralLayer(obj, platform);
        end
        
        % propagate forward
        function [x] = forwProp(obj, z)
            
            x = max(0, z);
            obj.x = x;
        end
        
        % propagate backward
        function [dydz] = backProp(obj, dydx)
            
            obj.num_out = size(dydx, 2);
            obj.dxdz = obj.repout2(double(obj.x > 0));
            dydz = dydx .* obj.dxdz;
        end
        
        % R{propagate forward}
        function [Rx] = RforwProp(obj, Rz)
            
            Rx = obj.dxdz .* Rz;
            obj.Rx = Rx;
        end
        
        % R{propagate backward}
        function [Rdydz] = RbackProp(obj, Rdydx)
            
            Rdydz = Rdydx .* obj.dxdz;
        end
        
    end
end
