function y = vl_nnpoolangle(x,varargin)

dzdy = [];
if nargin > 1
    if ~ischar(varargin{1})
        dzdy = varargin{1};
        if numel(varargin) > 1
            varargin = varargin(2:end);
        end
    end
end

opts.bins = 1;
opts.angle_n = 8;
opts.max_angle = 360;
opts.output_relative_angles = false;
opts.output_absolute_angles = false;
opts.opts = {};
if isempty(varargin{end})
    varargin = varargin(1:end-1);
end
opts = vl_argparse(opts, varargin);

bins = opts.bins;
angle_n = opts.angle_n;
max_angle = opts.max_angle;
rel_ang = opts.output_relative_angles;
abs_ang = opts.output_absolute_angles;

if bins == 0
    output_mode = 2;
    bins = 1;
    rel_ang = false;
    abs_ang = false;
else
    output_mode = 1;
end

if rel_ang || abs_ang
    bins = 1;
end

if nargin <= 3 || isempty(dzdy) % Forward
    x = permute(x,[1 2 4 3]);
    xs = stackSteeredFilters(x,angle_n);
    nearest_center = getNearestCenter(angle_n,bins,max_angle);
    if output_mode == 1
        if abs_ang || rel_ang
            y = zeros(size(xs,1),size(xs,2),size(xs,3),...
                size(xs,4)*(1 + 2*abs_ang + 2*(size(xs,4)-1)*rel_ang),1,'like',xs);
            [y(:,:,:,1:size(xs,4)), y_ind] = max(xs,[],5);
            y_ind = y_ind*2*pi/angle_n;
            b = 1;
            if rel_ang
                for b = 1:size(xs,4)-1
                    dif_y = y_ind - circshift(y_ind,b,4);
                    y(:,:,:,(2*b-1)*size(xs,4)+1:2*b*size(xs,4)) = cos(dif_y);
                    y(:,:,:,2*b*size(xs,4)+1:(2*b+1)*size(xs,4)) = sin(dif_y);
                end
                b = b + 1;
            end
            if abs_ang
                y(:,:,:,(2*b-1)*size(xs,4)+1:2*b*size(xs,4)) = cos(y_ind);
                y(:,:,:,2*b*size(xs,4)+1:(2*b+1)*size(xs,4)) = sin(y_ind);
            end
        else
            y = zeros(size(xs,1),size(xs,2),size(xs,3),size(xs,4),bins,'like',xs);
            for i = 1:bins
                y(:,:,:,:,i) = max(xs(:,:,:,:,nearest_center==i),[],5);
            end
        end
        y = stackSteeredFilters(y,bins);
        y = permute(y,[1 2 4 3]);
    else % output_mode == 2
        d_alpha = max_angle/angle_n;
        alphas = 0:d_alpha:max_angle;
        alphas(end) = [];
        p1 = cos(alphas*pi/180);
        p2 = sin(alphas*pi/180);
        [absy,inds] = max(xs,[],5);
        inds = reshape(inds,size(absy));
        y1 = absy .* reshape(p1(inds),size(absy));
        y2 = absy .* reshape(p2(inds),size(absy));
        %y1 = stackSteeredFilters(y1,bins);
        y1 = permute(y1,[1 2 4 3]);
        %y2 = stackSteeredFilters(y2,bins);
        y2 = permute(y2,[1 2 4 3]);
        y = zeros([size(y1,1),size(y1,2),size(y1,3),size(y1,4),2],'like',x);
        y(:,:,:,:,1) = y1;
        y(:,:,:,:,2) = y2;
        %y = stackSteeredFilters(permute(y,[1 2 4 3 5]),2);
        %y = permute(y,[1 2 4 3]);
    end
else % Backward
    if abs_ang || rel_ang
        dzdy = dzdy(:,:,1:size(x,3)/angle_n,:,:);
    end
    x = permute(x,[1 2 4 3]);
    xs = stackSteeredFilters(x,angle_n);
    nearest_center = getNearestCenter(angle_n,bins,max_angle);
    
    if output_mode == 2
        %temp_dzdy = zeros(size(dzdy(:,:,:,:,1)),'like',dzdy);
        %for m = 1:2
        %    temp_dzdy = temp_dzdy + dzdy(:,:,:,:,m).^2;
        %end
        %dzdy = permute(dzdy,[1 2 4 3]);
        %dzdy = stackSteeredFilters(dzdy,2);
        %temp_dzdy = sqrt(sum(dzdy.^2,5));
        d_alpha = opts.max_angle/angle_n;
        alphas = 0:d_alpha:opts.max_angle;
        alphas(end) = [];
        p1 = cos(alphas*pi/180);
        p2 = sin(alphas*pi/180);
        [absy,inds] = max(xs,[],5);
        inds = permute(inds,[1 2 4 3]);
        dzdy = dzdy(:,:,:,:,1).*reshape(p1(inds),size(inds)) + dzdy(:,:,:,:,2).*reshape(p2(inds),size(inds)) ;
        %dzdy = dzdy(:,:,:,:,1) + dzdy(:,:,:,:,2);
        %dzdy = (temp_dzdy).*sign(permute(absy,[1 2 4 3]));
        %inds = permute(inds,[1 2 4 3]);
        %dzdy = dzdy(:,:,:,:,1).*sign(p1(inds)) + dzdy(:,:,:,:,2).*sign(p2(inds));
    end
    
    dzdy = permute(dzdy,[1 2 4 3]);
    dzdys = stackSteeredFilters(dzdy,bins);
    y = zeros(size(xs,1),size(xs,2),size(xs,3),size(xs,4),angle_n,'like',xs);
    
    inds = zeros(size(xs,1),size(xs,2),size(xs,3),size(xs,4),bins,'like',xs);
    inds_pre = zeros(size(xs,1),size(xs,2),size(xs,3),size(xs,4),bins,'like',xs);
    
    offset = 0;
    for i = 1:bins
        [~,inds(:,:,:,:,i)] = max(xs(:,:,:,:,nearest_center==i),[],5);
        inds_pre(:,:,:,:,i) = inds(:,:,:,:,i) + offset;
        offset = offset + sum(nearest_center==i);
    end
    
    
    for i = 1:bins
        new_y = zeros(size(y,1),size(y,2),size(y,3),size(y,4),sum(nearest_center==i),'like',y);
        for j = 1:sum(nearest_center==i)
            new_y(:,:,:,:,j) = dzdys(:,:,:,:,i).*(inds(:,:,:,:,i)==j);
        end
        y(:,:,:,:,nearest_center==i) = new_y;
    end
    
   
    y = stackSteeredFilters(y,angle_n);
    y = permute(y,[1 2 4 3]);

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nearest_center = getNearestCenter(angle_n,bins,max_angle)

bin_centers = linspace(0,max_angle,bins+1);
bin_centers = bin_centers(1:end-1);
angles = linspace(0,max_angle,angle_n+1);
angles = angles(1:end-1);
dist_to_centers = pdist2(bin_centers',angles');
[~,nearest_center] = min(dist_to_centers,[],1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fs = stackSteeredFilters(f,angle_n)

siz = [size(f),1,1];

if size(f,5) == 1 % it's a normal 4D filter bank
    filt_num = siz(4) / angle_n;
    fs = zeros([siz(1:3), filt_num, angle_n],'like',f);
    for i = 1:angle_n
        fs(:,:,:,:,i) = f(:,:,:,i:angle_n:filt_num*angle_n-angle_n + i);
    end
else % it's a steered 5D filter bank
    angle_n = siz(5);
    filt_num = angle_n * siz(4);
    fs = zeros([siz(1:3), filt_num],'like',f);
    for i = 1:angle_n
        fs(:,:,:,i:angle_n:filt_num-angle_n + i) = f(:,:,:,:,i);
    end
end
end