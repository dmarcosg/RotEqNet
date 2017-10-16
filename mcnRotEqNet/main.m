setup_mcnRotEqNet

%% Downloads the MNIST-rot dataset and trains a rotation invariant classifier
opts = struct();
opts.train.gpus = 1; %opts.train.gpus = []; if no GPU available
cnn_mnist_rot_dag(opts);

%% Loads the test data and computes the model accuracy
cnn_mnist_experiments