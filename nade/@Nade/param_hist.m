function param_hist(obj)
% Displays a histogram of all parameters of nade.
% 
% George Papamakarios, Jun 2015

% hidden layer
figure;
suptitle('Hidden layer');

subplot(1,2,1);
nbins = floor(sqrt(numel(obj.W)));
hist(obj.W(:), nbins);
title('W');

subplot(1,2,2);
nbins = floor(sqrt(numel(obj.c)));
hist(obj.c(:), nbins);
title('c');

% output layer
figure;
suptitle('Output layer');

subplot(1,2,1);
nbins = floor(sqrt(numel(obj.U)));
hist(obj.U(:), nbins);
title('U');

subplot(1,2,2);
nbins = floor(sqrt(numel(obj.b)));
hist(obj.b(:), nbins);
title('b');
