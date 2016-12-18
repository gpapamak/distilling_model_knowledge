function changePlatform(obj, platform)
% Changes the platform to the one specified.
% INPUT
%     platform   either 'cpu' or 'gpu'
% 
% George Papamakarios, Jul 2015

switch platform
    
    case 'cpu'
        
        if ~obj.gpu
            return;
        end
        
        obj.gpu = false;
        obj.arraytype = 'double';
        f = @gather;
        
    case 'gpu'
        
        if obj.gpu
            return;
        end
        
        assert(gpuDeviceCount() > 0, 'No gpu found.');
        
        obj.gpu = true;
        obj.arraytype = 'gpuArray';
        f = @gpuArray;
        
    otherwise
        error('Unknown platform.');
end

if isfield(obj.layers{1}, 'x')
    obj.layers{1}.x = f(obj.layers{1}.x);
end
if isfield(obj.layers{1}, 'dydx')
    obj.layers{1}.dydx = f(obj.layers{1}.dydx);
end
if isfield(obj.layers{1}, 'Rx')
    obj.layers{1}.Rx = f(obj.layers{1}.Rx);
end

for i = 1:obj.num_layers
    obj.weights{i} = f(obj.weights{i});
    obj.biases{i}  = f(obj.biases{i});
    obj.layers{i+1}.changePlatform(platform);
end

obj.params = f(obj.params);
obj.fixed  = f(obj.fixed);

obj.dydp = f(obj.dydp);
obj.Hpvx = f(obj.Hpvx);
obj.Hxvx = f(obj.Hxvx);
