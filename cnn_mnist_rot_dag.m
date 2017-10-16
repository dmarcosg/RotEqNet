function [net, info] = cnn_mnist_rot_dag(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
run(fullfile(fileparts(mfilename('fullpath')),...
   'setup_mcnRotEqNet.m')) ;

opts.batchNormalization = true ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('models', ['mnist-rot_size5_12k']) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = 'data';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = netinit_mnist_dag() ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

trainfn = @cnn_train_dag ;

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3));

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;



function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
for i = 1:size(images,4)
    images(:,:,:,i) = imrotate(images(:,:,:,i),rand*360,'crop');
end
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;


% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Prepare the imdb structure, returns image data with mean image subtracted
urlroot = 'http://www.iro.umontreal.ca/~lisa/icml2007data/';
files = {'mnist_rotation_new.zip'};
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

if ~exist(fullfile(opts.dataDir, files{1}), 'file')
    url = sprintf([urlroot,files{1}]) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
    unzip(fullfile(opts.dataDir,files{1}),opts.dataDir);
end

d = single(dlmread(fullfile(opts.dataDir,'mnist_all_rotation_normalized_float_train_valid.amat')));

data = reshape(d(:,1:end-1)',28,28,1,[]);

set = [ones(1,size(data,4)-100) 3*ones(1,100)];
y = d(:,end)';

imdb.images.data = data ;
imdb.images.labels = y + 1 ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
