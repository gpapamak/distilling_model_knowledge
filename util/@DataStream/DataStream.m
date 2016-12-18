classdef DataStream < handle
% Abstract class. Specifies the interface of a data stream. 
% The user can request from the stream to generate a new data batch of a
% specified size. Useful for online learning.
%
% George Papamakarios, Feb 2015
    
    properties (SetAccess = protected, GetAccess = public)
        
        gen_derivs = false
        
    end

    methods (Abstract, Access = public)
        
        % generates a new data batch of size N
        [varargout] = gen(obj, N)
        
    end
    
    methods (Access = public)
        
        % determines whether to generate derivatives;
        % note that subclasses are allowed to ignore this, but it needs to
        % be there for uniformity with the subclasses that actually use it
        function genDerivs(obj, flag)
            obj.gen_derivs = flag;
        end
        
    end
end
