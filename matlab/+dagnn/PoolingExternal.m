classdef PoolingExternal < dagnn.Filter
  properties
    poolSize = [1 1]
  end

  methods
    function outputs = forward(self, inputs, params)
        if numel(inputs) == 2
            outputs{1} = vl_nnpool_ext(inputs{1},inputs{2}, self.poolSize, ...
                'pad', self.pad) ;
        else
            if size(inputs{1},5) == 1
                outputs{1} = vl_nnpool_ext(inputs{1},inputs{1}, self.poolSize,'pad', self.pad) ;
            end
            if size(inputs{1},5) == 2
                nor = sqrt(inputs{1}(:,:,:,:,1).^2 + inputs{1}(:,:,:,:,2).^2);
                res1 = vl_nnpool_ext(inputs{1}(:,:,:,:,1),nor, self.poolSize,'pad', self.pad) ;
                res2 = vl_nnpool_ext(inputs{1}(:,:,:,:,2),nor, self.poolSize,'pad', self.pad) ;
                outputs{1} = cat(5,res1,res2);
            end
        end
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        if numel(inputs) == 2
            derInputs{1} = vl_nnpool_ext(inputs{1}, self.poolSize, derOutputs{1}, ...
                               'pad', self.pad) ;
        end
        
        if numel(inputs) == 2
            derInputs{1} = vl_nnpool_ext(inputs{1},inputs{2}, self.poolSize, ...
                derOutputs{1},'pad', self.pad) ;
        else
            if size(inputs{1},5) == 1
                derInputs{1} = vl_nnpool_ext(inputs{1},inputs{1}, self.poolSize,derOutputs{1},'pad', self.pad) ;
            end
            if size(inputs{1},5) == 2
                nor = sqrt(inputs{1}(:,:,:,:,1).^2 + inputs{1}(:,:,:,:,2).^2);
                res1 = vl_nnpool_ext(inputs{1}(:,:,:,:,1),nor, self.poolSize,derOutputs{1}(:,:,:,:,1),'pad', self.pad) ;
                res2 = vl_nnpool_ext(inputs{1}(:,:,:,:,2),nor, self.poolSize,derOutputs{1}(:,:,:,:,2),'pad', self.pad) ;
                derInputs{1} = cat(5,res1,res2);
            end
        end
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = PoolingExternal(varargin)
      obj.load(varargin) ;
    end
  end
end
