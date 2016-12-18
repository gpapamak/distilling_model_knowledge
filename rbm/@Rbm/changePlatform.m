function changePlatform(obj, platform)
% Changes the platform to the one specified.
% INPUT
%     platform   either 'cpu' or 'gpu'
% 
% George Papamakarios, Jun 2015

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

obj.W = f(obj.W);
obj.a = f(obj.a);
obj.b = f(obj.b);

obj.gibbs_state = f(obj.gibbs_state);
