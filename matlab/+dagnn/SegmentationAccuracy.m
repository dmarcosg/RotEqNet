classdef SegmentationAccuracy < dagnn.Loss
    
    properties (Transient)
        pixelAccuracy = 0
        meanAccuracy = 0
        meanIntersectionUnion = 0
        confusion = 0
%         nCL  = 7;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % predictions = vl_nnsoftmax(inputs{1}) ;
            
            predictions = gather(inputs{1}) ;
            nCL = size(inputs{1},3);
            
            labels = gather(inputs{2}) ;
            % [~,predictions] = max(inputs{1}, [], 3) ;
            [~,predictions] = sort(predictions, 3, 'descend') ;
            predictions = predictions(:,:,1,:);
%             
%             legend = [255 255 255;   % 1  Impervious surfaces
%                 0 0 255;       % 2  Buildings
%                 0 255 255;     % 3  Low vegetation
%                 0 255 0;       % 4  Tree
%                 255 255 0;     % 5  Car
%                 255 0 0;       % 6  Clutter / background
%                 ]./255;
%             
%             figure; showImageArray(labels(:,:,1,:),'spacing',2,'normalize',0); title('labels');
%             colormap([0 0 0; legend])
%             figure; showImageArray(predictions(:,:,[1],:),'spacing',2,'normalize',0); title('predictions');
%             colormap([0 0 0; legend])
%             
            
            % compute statistics only on accumulated pixels
            ok = labels > 0 ;
            numPixels = sum(ok(:)) ;
            obj.confusion = obj.confusion + ...
                accumarray([labels(ok),predictions(ok)],1,[nCL nCL]) ;
            
            % compute various statistics of the confusion matrix
            pos = sum(obj.confusion,2) ;
            res = sum(obj.confusion,1)' ;
            tp = diag(obj.confusion) ;
            
            obj.pixelAccuracy = sum(tp) / sum(obj.confusion(:)+eps); % max(1,sum(obj.confusion(:))) ;
            obj.meanAccuracy = mean(tp ./ (pos+eps)); %max(1, pos)) ;
            obj.meanIntersectionUnion = mean(tp ./ (pos + res - tp + eps)); %max(1, pos + res - tp)) ;
            
            obj.average = [obj.meanAccuracy ; obj.pixelAccuracy ;  obj.meanIntersectionUnion] ;
            obj.numAveraged = obj.numAveraged + numPixels ;
            outputs{1} = obj.average ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = [] ;
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.confusion = 0 ;
            obj.pixelAccuracy = 0 ;
            obj.meanAccuracy = 0 ;
            obj.meanIntersectionUnion = 0 ;
            obj.average = [0;0;0] ;
            obj.numAveraged = 0 ;
        end
        
        function str = toString(obj)
            str = sprintf('acc:%.2f, mAcc:%.2f, mIU:%.2f', ...
                obj.pixelAccuracy, obj.meanAccuracy, obj.meanIntersectionUnion) ;
        end
        
        function obj = SegmentationAccuracy(varargin)
            obj.load(varargin) ;
        end
    end
end
