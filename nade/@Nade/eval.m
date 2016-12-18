function [L, y, dLdx] = eval(obj, x)
% Evaluates nade at specified input locations.
% INPUT
%     x     columns are input locations
% OUTPUT
%     L     log p(x)
%     y     p(xi|x<i) for all i 
%     dLdx  derivatives of log p(x) wrt x
% 
% George Papamakarios, Jun 2015

obj.forwProp(obj.precision(x(obj.fwd_order, :)));
L = obj.L;
y = obj.y(obj.rev_order, :);

if nargout > 2
    obj.backProp_inputs();
    dLdx = obj.dLdx(obj.rev_order, :);
end

obj.clear();
