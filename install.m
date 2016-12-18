% Adds all necessary paths to the system's search path.
% George Papamakarios, Jan 2015

addpath(fullfile(pwd, 'util'));
addpath(fullfile(pwd, 'pdfs'));
addpath(fullfile(pwd, 'samplers'));
addpath(fullfile(pwd, 'max_likelihood'));
addpath(fullfile(pwd, 'optimization'));
addpath(fullfile(pwd, 'neural_nets'));
addpath(fullfile(pwd, 'nade'));
addpath(fullfile(pwd, 'rbm'));
addpath(fullfile(pwd, 'expectation_propagation'));

addpath(fullfile(pwd, 'knowledge_distillation'));

addpath(fullfile(pwd, 'knowledge_distillation/bags_of_samples'));

addpath(fullfile(pwd, 'knowledge_distillation/neural_nets'));
addpath(fullfile(pwd, 'knowledge_distillation/neural_nets/mnist_allclass'));
addpath(fullfile(pwd, 'knowledge_distillation/neural_nets/mnist_binary'));

addpath(fullfile(pwd, 'knowledge_distillation/nade'));
