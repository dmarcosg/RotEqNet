classdef Sigmoid_1 < dagnn.ElementWise
  properties
    apply = []
  end
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnsigmoid_1(inputs{1},'apply',obj.apply) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnsigmoid_1(inputs{1}, derOutputs{1},'apply',obj.apply) ;
      derParams = {} ;
    end
    
    function obj = Sigmoid_1(varargin)
      obj.load(varargin) ;
    end
  end
end
