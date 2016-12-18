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

obj.fwd_order = f(obj.fwd_order);
obj.rev_order = f(obj.rev_order);

obj.W = f(obj.W);
obj.c = f(obj.c);
obj.U = f(obj.U);
obj.b = f(obj.b);

obj.x = f(obj.x);
obj.h = f(obj.h);
obj.y = f(obj.y);
obj.L = f(obj.L);

obj.dLdW = f(obj.dLdW);
obj.dLdc = f(obj.dLdc);
obj.dLdU = f(obj.dLdU);
obj.dLdb = f(obj.dLdb);

obj.dLdx = f(obj.dLdx);
obj.dLda = f(obj.dLda);

obj.RdLdW = f(obj.RdLdW);
obj.RdLdc = f(obj.RdLdc);
obj.RdLdU = f(obj.RdLdU);
obj.RdLdb = f(obj.RdLdb);
