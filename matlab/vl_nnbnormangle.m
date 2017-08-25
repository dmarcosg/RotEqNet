function [y,dzdg,dzdb,dzdmoments] = vl_nnbnormangle(x,g,b,varargin)


if isa(x,'gpuArray')
    g = gpuArray(g);
    b = gpuArray(b);
end


dzdy = [];
if nargin > 1 && ~isempty(varargin)
    if ~ischar(varargin{1})
        dzdy = varargin{1};
        if numel(varargin) > 1
            varargin = varargin(2:end);
        else
            varargin = [];
        end
    end
end

opts.moments = [];
opts.epsilon = 1e-9;
if ~isempty(varargin)
    opts = vl_argparse(opts, varargin);
end

if isempty(opts.moments)
    testMode = false;
else
    testMode = true;
    opts.moments(:,1) = 0;
    if isa(x,'gpuArray')
        opts.moments = gpuArray(opts.moments);
    end
end

%%%%%%%%%%%%% REMOVE THIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testMode = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_size = [size(x,1), size(x,2), size(x,3), size(x,4),size(x,5)] ;
g_size = size(g) ;
b_size = size(b) ;
g = reshape(g, [1 x_size(3) 1]) ;
b = reshape(b, [1 x_size(3) 1]) ;
xr = [];
% for i = 1:2
%     xr = cat(3,xr,reshape(x(:,:,:,:,i), [x_size(1)*x_size(2) x_size(3) x_size(4)])) ;
% end

xr = reshape(x,[x_size(1)*x_size(2) x_size(3) x_size(4)*2]);

mass = 2*prod(x_size([1 2 4])) ;
sigma2 = sum(sum(xr.^2,1),3) / mass + opts.epsilon ;
sigma = sqrt(sigma2) ;


if isempty(dzdy) %%%% FORWARD
    
    if testMode
        y = bsxfun(@times, g ./ opts.moments(:,2)', xr);
    else
        y = bsxfun(@times, g ./ sigma, xr);
    end
    
    dzdg = [];
    dzdb = [];
    dzdmoments = [];
    
    
else %%%%%%% BACKWARD
    
    dzdy = reshape(dzdy, size(xr)) ;
    dzdg = sum(sum(dzdy .* xr,1),3) ./ sigma ;
    dzdb = zeros(size(dzdg),'like',x);


    y = ...
    bsxfun(@times, g ./ sigma, dzdy) - ...
    bsxfun(@times, g .* dzdg ./ (sigma2 * mass), xr) ;

    dzdg = reshape(dzdg, g_size) ;
    dzdb = reshape(dzdb, b_size) ;
    dzdmoments = zeros([numel(g),2],'like',x);
    dzdmoments(:,2) = sigma';
    

end
y = reshape(y, size(x)) ;
end
