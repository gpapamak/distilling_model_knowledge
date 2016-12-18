classdef LogisticLayer < NeuralLayer
% Logistic neural network layer.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        dydx = []
        dxdz = []
        
        thres = 36
    end

    methods (Access = public)
        
        % constructor
        function [obj] = LogisticLayer(num_units, arraytype)
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
            
            % limit input to avoid underflow
            z(z >  obj.thres) =  obj.thres;
            z(z < -obj.thres) = -obj.thres;
            x = 1 ./ (1 + exp(-z));
            obj.x = x;
        end
        
        % propagate backward
        function [dydz] = backProp(obj, dydx)
            
            obj.num_out = size(dydx, 2);
            obj.dydx = dydx;
            obj.dxdz = obj.repout2(obj.x .* (1 - obj.x));
            dydz = dydx .* obj.dxdz;
        end
        
        % R{propagate forward}
        function [Rx] = RforwProp(obj, Rz)
            
            Rx = obj.dxdz .* Rz;
            obj.Rx = Rx;
        end
        
        % R{propagate backward}
        function [Rdydz] = RbackProp(obj, Rdydx)
            
            Rdxdz = obj.Rx .* obj.repout2(1 - 2 * obj.x);
            Rdydz = Rdydx .* obj.dxdz + obj.dydx .* Rdxdz;
        end
        
    end
end
