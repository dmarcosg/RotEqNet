% Wrapper for BilinearSampler block:
% (c) 2016 Ankush Gupta

classdef BilinearSampler < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
      is_vector = (size(inputs{1},5) == 2);
      if is_vector
          inputs{1} = cat(3,inputs{1}(:,:,:,:,1),inputs{1}(:,:,:,:,2));
      end
      outputs = vl_nnbilinearsampler(inputs{1}, inputs{2});
      if is_vector
          outputs = cat(5,outputs(:,:,1:size(outputs,3)/2,:,:),outputs(:,:,size(outputs,3)/2+1:end,:,:));
      end
      outputs = {outputs};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      is_vector = (size(inputs{1},5) == 2);
      if is_vector
          inputs{1} = cat(3,inputs{1}(:,:,:,:,1),inputs{1}(:,:,:,:,2));
          derOutputs{1} = cat(3,derOutputs{1}(:,:,:,:,1),derOutputs{1}(:,:,:,:,2));
      end
      [dX,dG] = vl_nnbilinearsampler(inputs{1}, inputs{2}, derOutputs{1});
      if is_vector
          dX = cat(5,dX(:,:,1:size(dX,3)/2,:,:),dX(:,:,size(dX,3)/2+1:end,:,:));
      end
      derInputs = {dX,dG};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      xSize = inputSizes{1};
      gSize = inputSizes{2};
      outputSizes = {[gSize(2), gSize(3), xSize(3), xSize(4)]};
    end

    function obj = BilinearSampler(varargin)
      obj.load(varargin);
    end
  end
end
