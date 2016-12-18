function clear(obj)
% Clears nade of all intermediate results.
%
% George Papamakarios, Jun 2015

obj.x = [];
obj.h = [];
obj.y = [];
obj.L = [];

obj.dLdW = [];
obj.dLdc = [];
obj.dLdU = [];
obj.dLdb = [];
        
obj.dLdx = [];
obj.dLda = [];

obj.RdLdW = [];
obj.RdLdc = [];
obj.RdLdU = [];
obj.RdLdb = [];

obj.done_forwProp = false;
obj.done_backProp = false;
