function changeLayerType(obj, layer, newtype)
% Changes the nonlinearity of a specified layer.
% INPUT
%     layer        layer to change(or 'final' for final layer)
%     newtype      type of new nonlinearity
% 
% George Papamakarios, Apr 2015

% parse input
p = inputParser;
p.addRequired('layer', @(t) (isscalar(t) && isint(t) && 1 <= t && t <= obj.num_layers) || any(validatestring(t, {'final'})));
p.addRequired('newtype', @(t) any(validatestring(t, {'logistic', 'probit', 'linear', 'relu', 'softmax', 'logsoftmax'})));
p.parse(layer, newtype);

% if last layer is specified
if strcmp(layer, 'final')
    layer = obj.num_layers;
end

% set nonlinearity
layer = layer + 1;
num_units = obj.num_units(layer);
switch newtype
    case 'logistic'
        obj.layers{layer} = LogisticLayer(num_units, obj.arraytype);
    case 'probit'
        obj.layers{layer} = ProbitLayer(num_units, obj.arraytype);
    case 'linear'
        obj.layers{layer} = LinearLayer(num_units, obj.arraytype);
    case 'relu'
        obj.layers{layer} = ReluLayer(num_units, obj.arraytype);
    case 'softmax'
        obj.layers{layer} = SoftmaxLayer(num_units, obj.arraytype);
    case 'logsoftmax'
        obj.layers{layer} = LogSoftmaxLayer(num_units, obj.arraytype);
end

% propagation now has to be done again
obj.done_forwProp = false;
obj.done_backProp = false;
