classdef PoolingAngle < dagnn.Filter
  properties
    bins = 1
    angle_n = 8
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnpoolangle(inputs{1}, ...
                             'bins', self.bins, ...
                             'angle_n', self.angle_n) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnpoolangle(inputs{1}, derOutputs{1}, ...
                               'bins', self.bins, ...
                               'angle_n', self.angle_n) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = [1 1] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = PoolingAngle(varargin)
      obj.load(varargin) ;
    end
  end
end
