function y = vl_nnpool_ext(x,ext,pool,varargin)

% Check whether forward or backward
dzdy = [];
if nargin > 3
    if ~ischar(varargin{1})
        dzdy = varargin{1};
        if numel(varargin) > 1
            varargin = varargin(2:end);
        else
            varargin = [];
        end
    end
end

opts.pad = 0 ;
%opts.stride = 1 ;
if ~isempty(varargin)
    opts = vl_argparse(opts, varargin);
end

pad = opts.pad;

if numel(pad) == 1
    pad = [ceil(pad/2) floor(pad/2) ceil(pad/2) floor(pad/2)];
end

if numel(pad) == 2
    pad = [ceil(pad(1)/2) floor(pad(1)/2) ceil(pad(2)/2) floor(pad(2)/2)];
end

if mod(size(x,1),pool) ~= 0
    pad = pad + [ceil(mod(size(x,1),pool)/2), floor(mod(size(x,1),pool)/2),...
        ceil(mod(size(x,1),pool)/2), floor(mod(size(x,1),pool)/2)];
end


ingpu = isa(x,'gpuArray');
if ingpu
    x = gather(x);
    ext = gather(ext);
    dzdy = gather(dzdy);
end


if nargin < 3 || isempty(dzdy) % Forward
    
    padded_ext = zeros(size(ext,1) + pad(1) + pad(2),size(ext,2) + pad(3) + pad(4),size(ext,3),size(ext,4),size(ext,5),'like',ext);
    padded_ext(pad(1)+1:end - pad(2), pad(3) + 1:end - pad(4),:,:,:,:) = ext;
    padded_x = zeros(size(x,1) + pad(1) + pad(2),size(x,2) + pad(3) + pad(4),size(x,3),size(x,4),size(x,5),'like',x);
    padded_x(pad(1)+1:end - pad(2), pad(3) + 1:end - pad(4),:,:,:,:) = x;
    [~,ind] = maxpool(padded_ext,pool);
    y = getmax(padded_x,pool,ind);
    %y = zeros(size(temp,1) + pad(1) + pad(2),size(temp,2) + pad(3) + pad(4),size(temp,3),size(temp,4),size(temp,5),'like',temp);
    %y(pad(1)+1:end - pad(2), pad(3) + 1:end - pad(4),:,:,:,:) = temp;
    
else % Backward
    padded_ext = zeros(size(ext,1) + pad(1) + pad(2),size(ext,2) + pad(3) + pad(4),size(ext,3),size(ext,4),size(ext,5),'like',ext);
    padded_ext(pad(1)+1:end - pad(2), pad(3) + 1:end - pad(4),:,:,:,:) = ext;
    [~,ind] = maxpool(padded_ext,pool);
    y = maxpool_back(dzdy,pool,size(padded_ext),ind);
    y = y(pad(1)+1:end - pad(2), pad(3) + 1:end - pad(4),:,:,:,:);
    
end

if ingpu
    y = gpuArray(y);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m,ind] = maxpool(x,pool)

if numel(pool) == 1
    pool = [pool pool];
end

x_pad = -inf(ceil(size(x,1)/pool(1))*pool(1),ceil(size(x,2)/pool(2))*pool(2),size(x,3),size(x,4),'like',x);
x_pad(1:size(x,1),1:size(x,2),:,:) = x;

m = zeros((size(x_pad,1)/pool(1)),(size(x_pad,2)/pool(2)),size(x_pad,3),size(x_pad,4),'like',x_pad);
ind = zeros(size(m),'like',m);

col = my_im2col(x_pad,pool,'distinct');
[temp_m,temp_ind] = max(col);
m = reshape(temp_m,size(m,1),size(m,2),size(x_pad,3),size(x_pad,4));
ind = reshape(temp_ind,size(m,1),size(m,2),size(x_pad,3),size(x_pad,4));


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = maxpool_back(m,pool,s,ind)

if numel(pool) == 1
    pool = [pool pool];
end


col_size = prod(pool);
num_cols = size(m,1)*size(m,2);


col = zeros(col_size,num_cols,size(m,3),size(m,4),'like',m);
offset = (0:numel(ind)-1)*prod(pool);
%offset = reshape(offset,size(ind));
col(ind(:) + offset(:)) = m(:);
x = my_col2im(col,pool,s,'distinct');

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m] = getmax(x,pool,ind)

if numel(pool) == 1
    pool = [pool pool];
end

x_pad = -inf(ceil(size(x,1)/pool(1))*pool(1),ceil(size(x,2)/pool(2))*pool(2),size(x,3),size(x,4),'like',x);
x_pad(1:size(x,1),1:size(x,2),:,:) = x;

m = zeros((size(x_pad,1)/pool(1)),(size(x_pad,2)/pool(2)),size(x_pad,3),size(x_pad,4),'like',x_pad);


col = my_im2col(x_pad,pool,'distinct');
offset = (0:numel(ind)-1)*prod(pool);
offset = reshape(offset,size(ind));
m = reshape(col(ind+offset),size(ind));

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function col = my_im2col(im,p,mode)

s = size(im);
if numel(s) == 3
    s = [s 1];
end
if numel(s) == 2
    s = [s 1 1];
end
col = zeros(prod(p),ceil(s(1)/p(1))*ceil(s(2)/p(2)),s(3),s(4),'like',im);
init_patch = zeros(p(1),p(2),size(im,3),size(im,4),'like',im);
count = 1;

p1 = p(1);
p2 = p(2);
s1 = s(1);
s2 = s(2);
if isa(im,'gpuArray')
    s1 = gpuArray(s1);
    s2 = gpuArray(s2);
    p1 = gpuArray(p1);
    p2 = gpuArray(p2);
    count = gpuArray(1);
    one = gpuArray(1);
else
    one = 1;
end

first_corners = repmat( (1:p(1):s(1))',1,s(2)/p(2));
first_corners = bsxfun(@plus,first_corners,0:p(2)*s(1):s(1)*s(2)-1);
first_corners = first_corners(:);
first_corners = bsxfun(@plus, first_corners, 0:s(1)*s(2):s(1)*s(2)*s(3)-1);
first_corners = first_corners(:);
first_corners = bsxfun(@plus, first_corners, 0:s(1)*s(2)*s(3):s(1)*s(2)*s(3)*s(4)-1);
first_corners = first_corners(:);

for j = 0:p(2)-1
    for i = 0:p(1)-1
        temp = im(first_corners + i + s(1)*j);
        temp = reshape(temp,1,size(col,2),size(col,3),size(col,4));
        col(count,:,:,:) = temp;
        count = count + 1;
    end
end
% for j_start = one:p2:s2-one
%     for i_start = one:p1:s1-one
%         temp = init_patch;
%         temp2 = im(i_start:min(s1,i_start+p1-one),j_start:min(s2,j_start+p2-one),:,:);
%         temp(one:size(temp2,1),one:size(temp2,2),:,:) = temp2;
%         col(:,count,:,:) = reshape(temp,[],1,s(3),s(4));
%         count = count + one;
%     end
% end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im = my_col2im(col,p,s,mode)

if numel(s) == 3
    s = [s size(col,4)];
end
if numel(s) == 2
    s = [s size(col,3) size(col,4)];
end

im = zeros(s,'like',col);

count = 1;
if isa(im,'gpuArray')
    s = gpuArray(s);
    p = gpuArray(p);
    count = gpuArray(1);
end

first_corners = repmat( (1:p(1):s(1))',1,s(2)/p(2));
first_corners = bsxfun(@plus,first_corners,0:p(2)*s(1):s(1)*s(2)-1);
first_corners = first_corners(:);
first_corners = bsxfun(@plus, first_corners, 0:s(1)*s(2):s(1)*s(2)*s(3)-1);
first_corners = first_corners(:);
first_corners = bsxfun(@plus, first_corners, 0:s(1)*s(2)*s(3):s(1)*s(2)*s(3)*s(4)-1);
first_corners = first_corners(:);

for j = 0:p(2)-1
    for i = 0:p(1)-1
        temp = col(count,:,:,:);
        im(first_corners + i + s(1)*j) = temp(:);
        count = count + 1;
    end
end
% for j_start = 1:p(2):s(2)-1
%     for i_start = 1:p(1):s(1)-1
%         temp = col(:,count,:,:);
%         temp = reshape(temp,p(1),p(2),s(3),s(4));
%         this_p = [min(s(1),i_start+p(1)-1)-i_start+1,min(s(2),j_start+p(2)-1)-j_start+1];
%         im(i_start:min(s(1),i_start+p(1)-1),j_start:min(s(2),j_start+p(2)-1),:,:) = temp(1:this_p(1),1:this_p(2),:,:);
%         count = count + 1;
%     end
% end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m,ind] = maxpool2(x,pool)

if numel(pool) == 1
    pool = [pool pool];
end
%x_pad = zeros(size(x,1)+2*pad,size(x,2)+2*pad,size(x,3),size(x,4),'like',x);

m_temp = zeros(ceil(size(x,1)/pool(1)),size(x,2),size(x,3),size(x,4),'like',x);
ind_temp = zeros(ceil(size(x,1)/pool(1)),size(x,2),size(x,3),size(x,4),'like',x);
count = 1;
for i_start = 1:pool(1):size(x,1)-1
    i_end = min(i_start+pool(1),size(x,1));
    [m_temp(count,:,:,:),ind_temp(count,:,:,:)] = max(x(i_start:i_end,:,:,:),[],1);
    ind_temp(count,:,:,:) = ind_temp(count,:,:,:) + i_start - 1;
    count = count + 1;
end

m = zeros(ceil(size(x,1)/pool(1)),ceil(size(x,2)/pool(2)),size(x,3),size(x,4),'like',x);
ind1 = zeros(ceil(size(x,1)/pool(1)),ceil(size(x,2)/pool(2)),size(x,3),size(x,4),'like',x);
ind2 = zeros(ceil(size(x,1)/pool(1)),ceil(size(x,2)/pool(2)),size(x,3),size(x,4),'like',x);
count = 1;
for j_start = 1:pool(1):size(x,1)-1
    j_end = min(j_start+pool(2),size(x,2));
    [m(:,count,:,:),ind2(:,count,:,:)] = max(m_temp(:,j_start:j_end,:,:),[],2);
    %ind2(:,count,:,:) = ind2(:,count,:,:) + j_start - 1;
    count = count + 1;
end
end
