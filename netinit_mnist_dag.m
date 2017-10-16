function [net] = netinit_mnist_dag(varargin)

opts.useDropout   = false;
opts.useBnorm     = true;
opts.modelDir     = '.';
opts.weightDecay  = 1 ;
opts.imchannels   = 1;
opts.classWeights = [];
opts.warmstart    = false;
opts.loadnetfrom  = [];
opts = vl_argparse(opts, varargin) ;

opts.initBias = 0 ;

rng('default');
rng(0) ;

angle_n = 17;
lr = [1 0.5] ;
dropout_rate = 0.75;
leak = 0.0;

conv_filters          = zeros(1,5);

% BLOCK 1: CS1
conv_filters(end,:)   = [9,9,1,6,1];
%%%% MAX POOLING 
% BLOCK 2: CS2
conv_filters(end+1,:) = [9,9,6,16,2];
%%%% MAX POOLING 
% BLOCK 3: CS3
conv_filters(end+1,:) = [9,9,16,32,2];
conv_filters(end+1,:) = [1,1,32,128,1];
conv_filters(end+1,:) = [1,1,128,10,1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BLOCK 1: CS1
net = dagnn.DagNN() ; 
id = 1; nid = 1;
convBlock = dagnn.Convsteer('size', conv_filters(id,:), 'stride', 1,'angle_n',angle_n, 'hasBias', ...
    true, 'pad', floor(conv_filters(id,1)/2)) ;
net.addLayer(['convsteer' num2str(id)], convBlock, {'input'}, {['x' num2str(nid)]}, ...
    {['l' num2str(nid) 'f'], ['l' num2str(nid) 'b']}) ;
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).learningRate = lr(1);
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).weightDecay = opts.weightDecay;
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = lr(2);
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;
nid = nid + 1;
% --- RELU LAYER
net.addLayer(['relu' num2str(id)],dagnn.ReLU('leak',leak),['x' num2str(nid-1)],['x' num2str(nid)]); 
split_nid = nid;
nid = nid + 1;
% --- POOLANGLE LAYER
net.addLayer(['poolangle_v' num2str(id)],dagnn.PoolingAngle('bins',0,'angle_n',angle_n),['x' num2str(split_nid)],['x' num2str(nid)]); 
nid = nid + 1;
% --- POOL LAYER
poolBlock = dagnn.PoolingExternal('poolSize', 2, 'pad', [0 0 0 0]) ;
net.addLayer(['pool_v' num2str(id)], poolBlock,['x' num2str(nid-1)],['x' num2str(nid)]);%['x' num2str(nid)]); 
nid = nid + 1;
% --- BNORM LAYER
net.addLayer(['bnormangle' num2str(id)],dagnn.BatchNormAngle('numChannels',conv_filters(id,4)),...
    ['x' num2str(nid-1)],['x' num2str(nid)], ...
    {['l' num2str(nid) 'm'], ['l' num2str(nid) 'b'],['l' num2str(nid) 'x']}); 
nid = nid + 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BLOCK 2: CS2
id = id + 1; 
convBlock = dagnn.Convsteer('size', conv_filters(id,:), 'stride', 1,'angle_n',angle_n, 'hasBias', ...
    true, 'pad', floor(conv_filters(id,1)/2)) ;
net.addLayer(['convsteer' num2str(id)], convBlock, {['x' num2str(nid-1)]}, {['x' num2str(nid)]}, ...
    {['l' num2str(nid) 'f'], ['l' num2str(nid) 'b']}) ;
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).learningRate = lr(1);
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).weightDecay = opts.weightDecay;
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = lr(2);
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;
nid = nid + 1;

% --- RELU LAYER
net.addLayer(['relu' num2str(id)],dagnn.ReLU('leak',leak),['x' num2str(nid-1)],['x' num2str(nid)]); 
split_nid = nid;
nid = nid + 1;

% --- POOLANGLE LAYER
net.addLayer(['poolangle_v' num2str(id)],dagnn.PoolingAngle('bins',0,'angle_n',angle_n),['x' num2str(split_nid)],['x' num2str(nid)]); 
nid = nid + 1;
% --- POOL LAYER
poolBlock = dagnn.PoolingExternal('poolSize', 2, 'pad', [0 0 0 0]) ;
net.addLayer(['pool_v' num2str(id)], poolBlock,['x' num2str(nid-1)],['x' num2str(nid)]);%['x' num2str(nid)]); 
nid = nid + 1;
% --- BNORM LAYER
net.addLayer(['bnormangle' num2str(id)],dagnn.BatchNormAngle('numChannels',conv_filters(id,4)),...
    ['x' num2str(nid-1)],['x' num2str(nid)], ...
    {['l' num2str(nid) 'm'], ['l' num2str(nid) 'b'],['l' num2str(nid) 'x']}); 
nid = nid + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% FC1 steerable
id = id + 1; 
convBlock = dagnn.Convsteer('size', conv_filters(id,:), 'stride', 1,'angle_n',angle_n, 'hasBias', ...
    true, 'pad', 1) ;
net.addLayer(['convsteer' num2str(id)], convBlock, {['x' num2str(nid-1)]}, {['x' num2str(nid)]}, ...
    {['l' num2str(nid) 'f'], ['l' num2str(nid) 'b']}) ;
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).learningRate = lr(1);
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).weightDecay = opts.weightDecay;
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = lr(2);
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;
nid = nid + 1;

% --- POOLANGLE LAYER
net.addLayer(['poolangle' num2str(id)],dagnn.PoolingAngle('bins',1,'angle_n',angle_n),['x' num2str(nid-1)],['x' num2str(nid)]); 
nid = nid + 1;
% --- RELU LAYER
net.addLayer(['relu' num2str(id)],dagnn.ReLU('leak',leak),['x' num2str(nid-1)],['x' num2str(nid)]); 
nid = nid + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% FC2
id = id + 1; 
convBlock = dagnn.Conv('size', conv_filters(id,:)*diag([1 1 1 1 1]), 'stride', 1, 'hasBias', ...
    true, 'pad', 0) ;
net.addLayer(['conv' num2str(id)], convBlock, {['x' num2str(nid-1)]}, {['x' num2str(nid)]}, ...
    {['l' num2str(nid) 'f'], ['l' num2str(nid) 'b']}) ;
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).learningRate = lr(1);
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).weightDecay = opts.weightDecay;
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = lr(2);
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;
nid = nid + 1;
% --- RELU LAYER
net.addLayer(['relu' num2str(id)],dagnn.ReLU('leak',0.1),['x' num2str(nid-1)],['x' num2str(nid)]); 
nid = nid + 1;
% --- BNORM LAYER
net.addLayer(['bnorm' num2str(id)],dagnn.BatchNorm('numChannels',conv_filters(id,4)),...
    ['x' num2str(nid-1)],['x' num2str(nid)], ...
    {['l' num2str(nid) 'm'], ['l' num2str(nid) 'b'],['l' num2str(nid) 'x']}); 
nid = nid + 1;
% --- DROPOUT LAYER
dropoutBlock = dagnn.DropOut('rate', dropout_rate) ;
net.addLayer(['dropout' num2str(id)], dropoutBlock,['x' num2str(nid-1)],['x' num2str(nid)]);%['x' num2str(nid)]); 
nid = nid + 1;
% --- CONV LAYER
id = id + 1; 
convBlock = dagnn.Conv('size', conv_filters(id,:), 'stride', 1, 'hasBias', ...
    true, 'pad', 0) ;
net.addLayer(['conv' num2str(id)], convBlock, {['x' num2str(nid-1)]}, {'prediction'}, ...
    {['l' num2str(nid) 'f'], ['l' num2str(nid) 'b']}) ;
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).learningRate = lr(1);
net.params(net.getParamIndex(['l' num2str(nid) 'f'])).weightDecay = opts.weightDecay;
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = lr(2);
net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;

net.addLayer('objective', ...
    dagnn.Loss('loss', 'softmaxlog'), ...
    {'prediction', 'label'}, 'objective') ;
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
             
for nid = 1:numel(net.layers)        
    if strcmp(net.layers(nid).name(1:2), 'bn')
        net.params(net.getParamIndex(['l' num2str(nid) 'm'])).learningRate = 1;
        net.params(net.getParamIndex(['l' num2str(nid) 'b'])).learningRate = 1;
        net.params(net.getParamIndex(['l' num2str(nid) 'x'])).learningRate = 0.05;
        net.params(net.getParamIndex(['l' num2str(nid) 'm'])).weightDecay = 0;
        net.params(net.getParamIndex(['l' num2str(nid) 'b'])).weightDecay = 0;
    end
end
        
% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = [0.3*ones(1,10) 0.1*ones(1,10) 0.03*ones(1,10) 0.01*ones(1,10) 0.003*ones(1,10) 0.001*ones(1,10) 0.0003*ones(1,10)] ;
net.meta.trainOpts.weightDecay = 1e-2 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.trainOpts.batchSize = 600 ;

net.initParams();
