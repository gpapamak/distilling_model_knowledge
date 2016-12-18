function [x, L] = max(obj)
% Finds a sample that has high probability.
% OUTPUT
%     x     a high probability sample
%     L     its unnormalized log probability
% 
% George Papamakarios, Jun 2015

% -- find a minimum of the joint probability, hopfield net style

num_chains = 1000;

x = double(rand(obj.num_inputs, num_chains, obj.arraytype) > 0.5);
aa = obj.a * ones(1, num_chains, obj.arraytype);
bb = obj.b * ones(1, num_chains, obj.arraytype);

while true
    
    x_prev = x;
    
    h = obj.W' * x_prev + bb;
    h = double(h > 0);

    x = obj.W * h + aa;
    x = double(x > 0);
    
    if all(x(:) == x_prev(:))
        break;
    end    
end

% -- generate a lot of samples and keep the one with highest probability

N = 100 * num_chains;

gibbs_state = obj.gibbs_state;
obj.setGibbsState(x);

[x, L] = obj.gen(N);

obj.setGibbsState(gibbs_state);

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
