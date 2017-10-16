function [y,dzdf,dzdb] = vl_nnconvsteer(x,f,b,varargin)
% Forward: y = vl_nnconvsteer(x,f,b)
% Backward: [dzdx,dzdf,dzdb] = vl_nnconvsteer(x,f,b,dzdy)
% Options:
% 'angle_n':  number of rotations to compute for each filter in f

if isa(x,'gpuArray')
    f = gpuArray(f);
    b = gpuArray(b);
end

cudnn = 'CuDNN';
dzdy = [];
if nargin > 3
    if ~ischar(varargin{1})
        dzdy = varargin{1};
        if numel(varargin) > 1
            varargin = varargin(2:end); 
        else
            varargin = {};
        end
    end
    if ~isempty(varargin) && ischar(varargin{end})
        cudnn = varargin{end};
        varargin = varargin(1:end-1);
    end
end

opts.pad = 0 ;
opts.stride = 1 ;
opts.angle_n = 8;
opts.max_angle = 360;
%opts.
if ~isempty(varargin)
    opts = vl_argparse(opts, varargin);
end

if size(f,5) == 2
    input_mode = 2;
else
    input_mode = 1;
end

if opts.angle_n > 1
    mask = fspecial('disk', size(f,1)/2);
    mask = imresize(mask,[size(f,1),size(f,2)]);
    mask = mask / max(mask(:));
    mask = mask > 0.5;
    f = bsxfun(@times,f,mask);
end

if nargin <= 3 || isempty(dzdy) % Forward
    br = zeros(1,numel(b)*opts.angle_n,'like',b);
    for i = 1:numel(b)
        br((i-1)*opts.angle_n+1:i*opts.angle_n) = b(i);
    end
    
    % If each input pixel is a 2D vector, apply conv to each dimension
    % separately and add the result (scalar product)
    y = zeros(1,'like',b);
    fr = getSteeredFilters(f,opts.angle_n,opts.max_angle);
    for m = 1:input_mode
        tempy = vl_nnconv(x(:,:,:,:,m),fr(:,:,:,:,m),br,'pad',opts.pad,'stride',opts.stride,cudnn);
        y = y + tempy;
    end
    dzdf = [];
    dzdb = [];
    
else  % Backward
    
    % Multiplex the biases
    br = zeros(1,numel(b)*opts.angle_n,'like',b);
    for i = 1:numel(b)
        br((i-1)*opts.angle_n+1:i*opts.angle_n) = b(i);
    end
    b = br;
    

    dzdb_temp = {};
    dzdx_temp = {};
    
    % Get all steered filters
    fr = getSteeredFilters(f,opts.angle_n,opts.max_angle);
    dzdfr = zeros(size(fr),'like',fr);
    for m = 1:input_mode
        % Get gradients
        [dzdx_temp{m}, dzdfr(:,:,:,:,m), dzdb_temp{m}] = vl_nnconv(x(:,:,:,:,m),fr(:,:,:,:,m),b,dzdy,'pad',opts.pad,'stride',opts.stride,cudnn);
    
        % Demultiplex the bias gradients by averaging
        dzdbr = zeros(1,numel(dzdb_temp{m})/opts.angle_n,'like',dzdb_temp{m});
        for i = 1:numel(dzdbr)
            dzdbr(i) = sum(dzdb_temp{m}((i-1)*opts.angle_n+1:i*opts.angle_n));
        end
        dzdb_temp{m} = dzdbr;
    end
    % Un-steer the gradients by averaging
    dzdf = getSteeredFilters(dzdfr,-opts.angle_n,opts.max_angle);
    size_x = size(dzdx_temp{1});
    size_x(5) = input_mode;
    size_x(size_x==0) = 1;
    dzdx = zeros(size_x,'like',x);
    dzdb = zeros(size(dzdb_temp{1}),'like',x);
    for m = 1:input_mode
        dzdx(:,:,:,:,m) = dzdx_temp{m};
        dzdb = dzdb + dzdb_temp{m};
    end
    y = dzdx;% / sqrt(opts.angle_n);
    dzdb = dzdb(:);% / sqrt(opts.angle_n);% / input_mode;
    if opts.angle_n > 1
        dzdf = bsxfun(@times,dzdf,mask);
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fr = getSteeredFilters(f,angle_n,max_angle)
% wr = getRotatedFilters(w,alpha_n)
% Get alpha_n rotated versions of filters w
% f is a 4D array of 3D filters
% alpha_n >=1 is an integer, the number of steered versions

if angle_n < 0
    angle_n = abs(angle_n);
    inverse = true;
    fr = zeros([size(f,1),size(f,2),size(f,3),size(f,4)/angle_n,size(f,5)],'like',f);
else
    inverse = false;
    fr = zeros([size(f,1),size(f,2),size(f,3),size(f,4)*angle_n,size(f,5)],'like',f);
end

d_alpha = max_angle/angle_n;
alphas = 0:d_alpha:max_angle;
alphas(end) = [];


for i = 1:numel(alphas)
    if inverse
        fr = fr + rotateVectorField(f(:,:,:,i:angle_n:end,:),-alphas(i)); 
    else
        fr(:,:,:,i:angle_n:end,:) = rotateVectorField(f,alphas(i));
    end
end

end

