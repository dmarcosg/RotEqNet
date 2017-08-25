classdef DropOutInput < dagnn.ElementWise
  properties
    groups = 1
    prob = 1
    frozen = false
  end

  properties (Transient)
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
      if strcmp(obj.net.mode, 'test')
        outputs = inputs ;
        return ;
      end
      if obj.frozen & ~isempty(obj.mask)
        outputs{1} = vl_nndropout(inputs{1}, 'mask', obj.mask) ;
      else
        [outputs{1}, obj.mask] = vl_nndropout_input(inputs{1}, 'groups', obj.groups,'prob', obj.prob) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if strcmp(obj.net.mode, 'test')
        derInputs = derOutputs ;
        derParams = {} ;
        return ;
      end
      derInputs{1} = vl_nndropout_input(inputs{1}, derOutputs{1}, 'mask', obj.mask) ;
      derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = DropOutInput(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      %obj.groups = [];
      %obj.prob = [];
      %obj.frozen = false ;
    end
  end
end
