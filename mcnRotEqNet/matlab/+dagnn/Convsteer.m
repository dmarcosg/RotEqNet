classdef Convsteer < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
    angle_n = 8
    init_f = []
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconvsteer(...
        inputs{1}, params{1}, params{2}, ...
        'angle_n', obj.angle_n, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconvsteer(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'angle_n', obj.angle_n, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      if isempty(obj.init_f)
        sc = sqrt(2 / prod(obj.size(1:3))) ;
      else
        sc = obj.init_f;
      end
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 5 dimensions
      ksize = [ksize(:)' 1 1 1 1 1] ;
      obj.size = ksize(1:5) ;
    end

    function obj = Convsteer(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.angle_n =  obj.angle_n;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
