function setParamsFromVec(obj, params)
% Sets all the parameters of nade from those given as a long vector.
% 
% George Papamakarios, Jun 2015

i = 1;

% W
j = i + (obj.num_inputs - 1) * obj.num_hidden;
obj.W(:) = params(i:j-1);
i = j;

% c
j = i + obj.num_hidden;
obj.c = params(i:j-1)';
i = j;

% U
j = i + obj.num_inputs * obj.num_hidden;
obj.U(:) = params(i:j-1);
i = j;

% b
obj.b = params(i:end);
