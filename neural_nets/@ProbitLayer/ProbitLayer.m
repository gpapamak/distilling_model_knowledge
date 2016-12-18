classdef ProbitLayer < NeuralLayer
% Probit neural network layer.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        z    = []
        dydx = []
        dxdz = []
        
        thres = 8
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ProbitLayer(num_units, arraytype)
            obj@NeuralLayer(num_units, arraytype);
        end
        
        % clears intermediate results
        function clear(obj)
            obj.z    = [];
            obj.dydx = [];
            obj.dxdz = [];
            clear@NeuralLayer(obj);
        end
        
        % changes platform
        function changePlatform(obj, platform)
            switch platform
                case 'cpu'
                    obj.z    = gather(obj.z);
                    obj.dydx = gather(obj.dydx);
                    obj.dxdz = gather(obj.dxdz);
                case 'gpu'
                    obj.z    = gpuArray(obj.z);
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
            x = 0.5 * erf(z / sqrt(2)) + 0.5;
            obj.z = z;
            obj.x = x;
        end
        
        % propagate backward
        function [dydz] = backProp(obj, dydx)
            
            obj.num_out = size(dydx, 2);
            obj.dydx = dydx;
            obj.dxdz = obj.repout2(normpdf(obj.z));
            dydz = dydx .* obj.dxdz;
        end
        
        % R{propagate forward}
        function [Rx] = RforwProp(obj, Rz)
            
            Rx = obj.dxdz .* Rz;
            obj.Rx = Rx;
        end
        
        % R{propagate backward}
        function [Rdydz] = RbackProp(obj, Rdydx)
            
            Rdxdz = -obj.repout2(obj.z) .* obj.Rx;
            Rdydz = Rdydx .* obj.dxdz + obj.dydx .* Rdxdz;
        end
        
    end
end
