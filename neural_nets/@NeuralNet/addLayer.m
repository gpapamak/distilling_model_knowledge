function addLayer(obj, num_units, varargin)
% Adds a new layer to the network.
% INPUT
%     num_units    number of units of the new layer
% -- optional name-value pairs -- 
%     weights      matrix of weights
%     biases       vector of biases
%     fix_weights  if true, don't change weights in training
%     fix_biases   if true, don't change biases in training
%     type         type of the nonlinearity
% 
% George Papamakarios, Feb 2015

% if a loss function is set, adding layers is not allowed
assert(isempty(obj.loss_fun_hl), 'A loss function has been set. Adding more layers not allowed.');

% make the arrangements for the platform
if obj.gpu
    createArray = @gpuArray;
else
    createArray = @(x) x;
end

% number of units in the last layer
num_prev_units = obj.num_units(end);

% parse input
p = inputParser;
p.addRequired('num_units', @(t) isscalar(t) && isint(t) && t > 0);
p.addParameter('weights', 'random', @(t) (~ischar(t) && isreal(t) && isequal(size(t), [num_units, num_prev_units])) || any(validatestring(t, {'random'})));
p.addParameter('biases', 'random', @(t) (~ischar(t) && isreal(t) && isequal(size(t), [num_units, 1])) || any(validatestring(t, {'random'})));
p.addParameter('fix_weights', false, @(t) isscalar(t) && islogical(t));
p.addParameter('fix_biases', false, @(t) isscalar(t) && islogical(t));
p.addParameter('type', 'logistic', @(t) any(validatestring(t, {'logistic', 'probit', 'linear', 'relu', 'softmax', 'logsoftmax'})));
p.parse(num_units, varargin{:});

% add layer
obj.num_layers  = obj.num_layers + 1;
obj.num_units   = [obj.num_units, num_units];
obj.num_outputs = obj.num_units(end);
obj.num_params  = obj.num_params + (obj.num_units(end-1) + 1) * obj.num_outputs;

% set weights
if ischar(p.Results.weights)
    switch p.Results.weights
        case 'random'
            obj.weights = [obj.weights, {randn(num_units, num_prev_units, obj.arraytype) / sqrt(num_prev_units + 1)}];
    end
else
    obj.weights = [obj.weights, {createArray(p.Results.weights)}];
end
obj.params = [obj.params; obj.weights{end}(:)];
obj.fixed  = [obj.fixed;  createArray(p.Results.fix_weights & true(numel(obj.weights{end}), 1))];

% set biases
if ischar(p.Results.biases)
    switch p.Results.biases
        case 'random'
            obj.biases = [obj.biases, {randn(num_units, 1, obj.arraytype) / sqrt(num_prev_units + 1)}];
    end
else
    obj.biases = [obj.biases, {createArray(p.Results.biases)}];
end
obj.params = [obj.params; obj.biases{end}(:)];
obj.fixed  = [obj.fixed;  createArray(p.Results.fix_biases & true(numel(obj.biases{end}), 1))];

% set nonlinearity
switch p.Results.type
    case 'logistic'
        obj.layers = [obj.layers, {LogisticLayer(num_units, obj.arraytype)}];
    case 'probit'
        obj.layers = [obj.layers, {ProbitLayer(num_units, obj.arraytype)}];
    case 'linear'
        obj.layers = [obj.layers, {LinearLayer(num_units, obj.arraytype)}];
    case 'relu'
        obj.layers = [obj.layers, {ReluLayer(num_units, obj.arraytype)}];
    case 'softmax'
        obj.layers = [obj.layers, {SoftmaxLayer(num_units, obj.arraytype)}];
    case 'logsoftmax'
        obj.layers = [obj.layers, {LogSoftmaxLayer(num_units, obj.arraytype)}];
end

% propagation now has to be done again
obj.done_forwProp = false;
obj.done_backProp = false;
