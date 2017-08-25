%% Load and prepare test data
d = single(dlmread(fullfile('data','mnist_all_rotation_normalized_float_test.amat')));
data = reshape(d(:,1:end-1)',28,28,1,[]);
y = d(:,end)+1;

%% Get results from  the CNN softmax
bsize = 500;
pred = [];
count = 1;
net_names = {'models/mnist-rot_size5_12k/net-epoch-50.mat'};

angle_n = 17;
angs = 0;
%angs = [0:20:20*5]; Uncomment this line to perform 6x data augmentation
for i = 1:bsize:size(data,4)
    scores = zeros(1,1,10,bsize);
    for n = 1:max(1,numel(net_names))
        if ~isempty(net_names)
            load(net_names{n})
            net.layers{end}.type = 'softmax';
            for j = 1:numel(net.layers), if isfield(net.layers{j},'angle_n'), net.layers{j}.angle_n=angle_n; end, end
            net = vl_simplenn_tidy(net);
            net = vl_simplenn_move(net,'gpu');
        end
        for a = angs
            res = vl_simplenn(net,gpuArray(imrotate(data(:,:,:,i:min(size(data,4),i+bsize-1)),a,'bicubic','crop')),[],[],'mode','test');
            scores = scores + gather(res(end).x);
        end
    end
    [~,bpred] = max(scores,[],3);
    pred = [pred; bpred(:)];
    bfeat = res(end-2).x;
    disp([num2str(count) ' out of ' ceil(num2str(size(data,4)/bsize)), ': ', num2str(mean(y(1:numel(pred))~=pred(:)))]);
    count = count + 1;
end

disp(['Test error: ' num2str(mean(y~=pred(:))*100) '%'])

