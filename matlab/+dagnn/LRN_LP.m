classdef LRN_LP < dagnn.ElementWise
  properties
    epsilon = 1e-2
    p = 2
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1},[], 'epsilon',obj.epsilon,'p',obj.p) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, 'epsilon',obj.epsilon,'p',obj.p) ;
      derParams = {} ;
    end

    function obj = LRN_LP(varargin)
      obj.load(varargin) ;
    end
  end
end
