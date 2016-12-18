% regression demo of a feedforward neural net
% George Papamakarios, Feb 2015

clear;
close all;

% train data
x = randn(2, 1e+4);
y = sum(x .* x);

% create a net
net = NeuralNet(2);
net.addLayer(10, 'type', 'logistic');
net.addLayer( 5, 'type', 'logistic');
net.addLayer( 2, 'type', 'logistic');
net.addLayer( 1, 'type', 'linear');
net.setLossFunction('square_error');

% train the net
maxiter = 5000;
minibatch = 50;
step = LinearDecay(1, maxiter);
[~, ~, trace] = net.train(x, y, 'step', step, 'maxiter', maxiter, 'minibatch', minibatch);

% plot the learning progress
figure;
loglog(trace);
xlabel('iterations');
ylabel('square error');

% plot the learnt surface
figure; hold on;
xmin = -5;
xmax = 5;
xx = linspace(xmin, xmax);
[X, Y] = meshgrid(xx);
Z = reshape(net.eval([X(:), Y(:)]'), size(X));
surfc(X, Y, Z, 'facealpha', 0.4);
plot3(x(1,:), x(2,:), y, 'b.', 'MarkerSize', 10);
colorbar;
shading flat;
asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
axis([xmin xmax xmin xmax]);
xlabel('x_1');
ylabel('x_2');
zlabel('y');
