function [net, info] = cnn_mnist_rot(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST
addpath('~/Dropbox/Tools');
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = true ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('models', ['mnist-rot_size5_12k']) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

%opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.dataDir = 'data';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
%if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
opts.train.gpus = 1;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = cnn_mnist_init_size5('batchNormalization', opts.batchNormalization) ;

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

trainfn = @cnn_train ;

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y) ;


% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------

images = imdb.images.data(:,:,:,batch) ;
f = fspecial('gaussian',10,3);
for i = 1:size(images,4)
    images(:,:,:,i) = imrotate(images(:,:,:,i),rand*360,'crop');
    %warp = 3*imresize(convn(randn(10,10,2),f,'full'),size(images(:,:,1)));
    %warp = bsxfun(@plus,warp, cat(3,randn,randn)/4);
    %images(:,:,:,i) = imwarp(images(:,:,:,i),warp,'bicubic');
    %images(:,:,:,i) = imtranslate(images(:,:,:,i),[(rand-0.5)*1 (rand-0.5)*1],'cubic');
end


labels = imdb.images.labels(1,batch) ;


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
dataMean = mean(data(:,:,:,set == 1), 4);
%data = bsxfun(@minus, data, dataMean) ;
y = d(:,end)';

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = y + 1 ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
