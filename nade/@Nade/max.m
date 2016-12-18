function [x, L] = max(obj)
% Finds a sample that has high probability.
% OUTPUT
%     x     a high probability sample
%     L     its log probability
% 
% George Papamakarios, Jun 2015

% -- generate a lot of samples and keep the one with highest probability

N = 2000;

[y, x] = obj.gen(N);
L = sum(x .* log(y) + (1-x) .* log(1-y), 1);

[L, idx] = max(L);
x = x(:, idx);

% -- try to imrove on the best sample found by flipping bits

changed = true;

while changed
    
    changed = false;
    
    for i = 1:obj.num_inputs
        
        x(i) = 1 - x(i);
        Li = obj.eval(x);
        
        if Li < L
            x(i) = 1 - x(i);
        else
            changed = true;
            L = Li;
        end
    end
end
