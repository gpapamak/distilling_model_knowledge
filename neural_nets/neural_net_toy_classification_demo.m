% multiclass classification demo of a feedforward neural net
% George Papamakarios, Feb 2015

clear;
close all;

% train data
N_per_class = 1e+4;
x1 = 0.5*randn(2, N_per_class) + 2*[1; 0] * ones(1, N_per_class);
x2 = 0.5*randn(2, N_per_class) + 2*[-1/2; +sqrt(3)/2] * ones(1, N_per_class);
x3 = 0.5*randn(2, N_per_class) + 2*[-1/2; -sqrt(3)/2] * ones(1, N_per_class);
y1 = [1; 0; 0] * ones(1, N_per_class);
y2 = [0; 1; 0] * ones(1, N_per_class);
y3 = [0; 0; 1] * ones(1, N_per_class);
x = [x1 x2 x3];
y = [y1 y2 y3];

% create a net
net = NeuralNet(2);
net.addLayer(10, 'type', 'logistic');
net.addLayer( 5, 'type', 'logistic');
net.addLayer( 3, 'type', 'logsoftmax');
net.setLossFunction('dot_product');

% train the net
maxiter = 1000;
minibatch = 10;
step = LinearDecay(1, maxiter);
[~, ~, trace] = net.train(x, y, 'step', step, 'maxiter', maxiter, 'minibatch', minibatch);

% plot the learning progress
figure;
loglog(trace);
xlabel('iterations');
ylabel('cross entropy');

% plot the learnt surface
figure;
xmin = -5;
xmax = 5;
xx = linspace(xmin, xmax);
[X, Y] = meshgrid(xx);
Z = reshape(net.eval([X(:), Y(:)]')', [size(X), 3]);
Z = exp(Z);
for i = 1:3
    subplot(1,3,i); hold on;
    surfc(X, Y, Z(:,:,i), 'facealpha', 0.2);
    plot(x1(1,:), x1(2,:), 'b.', 'MarkerSize', 10);
    plot(x2(1,:), x2(2,:), 'r.', 'MarkerSize', 10);
    plot(x3(1,:), x3(2,:), 'g.', 'MarkerSize', 10);
    colorbar;
    shading flat;
    asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
    axis([xmin xmax xmin xmax]);
    xlabel('x_1');
    ylabel('x_2');
    zlabel(sprintf('P(y%d = 1)', i));
end
suptitle('The original net');

%% now compress it
net_comp = NeuralNet(2);
net_comp.addLayer(5, 'type', 'logistic');
net_comp.addLayer(3, 'type', 'logsoftmax');
%net_comp.setLossFunction('deriv_square_error');
net_comp.setLossFunction('cross_entropy_&_deriv_square_error', 0.1);

% train the net
maxiter = 5000;
minibatch = 10;
input_gen = @(N) gauss_sample([0; 0], 10 * eye(2), N);
[~, ~, trace] = net_comp.train_net(net, input_gen, 'maxiter', maxiter, 'minibatch', minibatch);

% plot the learning progress
figure;
loglog(trace);
xlabel('iterations');
ylabel('loss');

% plot the learnt surface
figure;
xmin = -5;
xmax = 5;
xx = linspace(xmin, xmax);
[X, Y] = meshgrid(xx);
Z = reshape(net_comp.eval([X(:), Y(:)]')', [size(X), 3]);
Z = exp(Z);
for i = 1:3
    subplot(1,3,i); hold on;
    surfc(X, Y, Z(:,:,i), 'facealpha', 0.2);
    plot(x1(1,:), x1(2,:), 'b.', 'MarkerSize', 10);
    plot(x2(1,:), x2(2,:), 'r.', 'MarkerSize', 10);
    plot(x3(1,:), x3(2,:), 'g.', 'MarkerSize', 10);
    colorbar;
    shading flat;
    asp_ratio = daspect; daspect([asp_ratio(1), asp_ratio(1), asp_ratio(3)]);
    axis([xmin xmax xmin xmax]);
    xlabel('x_1');
    ylabel('x_2');
    zlabel(sprintf('P(y%d = 1)', i));
end
suptitle('The compressed net');
