classdef LinearLayer < NeuralLayer
% Linear neural network layer.
%
% George Papamakarios, Mar 2015

    methods (Access = public)
        
        % constructor
        function [obj] = LinearLayer(num_units, arraytype)
            obj@NeuralLayer(num_units, arraytype);
        end 
        
        % propagate forward
        function [z] = forwProp(obj, z)
            obj.x = z;
        end
        
        % propagate backward
        function [dydx] = backProp(~, dydx)
        end
        
        % R{propagate forward}
        function [Rz] = RforwProp(obj, Rz)
            obj.Rx = Rz;
        end
        
        % R{propagate backward}
        function [Rdydx] = RbackProp(~, Rdydx)
        end
        
    end
end
