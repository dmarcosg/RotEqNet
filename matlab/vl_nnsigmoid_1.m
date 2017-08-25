function out = vl_nnsigmoid_1(x,varargin)
%VL_NNSIGMOID CNN sigmoid nonlinear unit.
%   Y = VL_NNSIGMOID(X) computes the sigmoid of the data X. X can
%   have an arbitrary size. The sigmoid is defined as follows:
%
%     SIGMOID(X) = 1 / (1 + EXP(-X)).
%
%   DZDX = VL_NNSIGMOID(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

% Copyright (C) 2015 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Check whether forward or backward
dzdy = [];
if nargin > 3
    if ~ischar(varargin{1})
        dzdy = varargin{1};
        if numel(varargin) > 1
            varargin = varargin(2:end);
        else
            varargin = [];
        end
    end
end

opts.apply = [];
opts = vl_argparse(opts, varargin) ;

if isempty(opts.apply)
    opts.apply = ones(1,size(x,3));
end

opts.apply = opts.apply > 0;

y = x;

y(:,:,opts.apply,:) = (1 ./ (1 + exp(-x(:,:,opts.apply,:))));

if nargin <= 1 || isempty(dzdy)
  out = y*2-1 ;
else
  out = dzdy;
  out(:,:,opts.apply,:) = 2*dzdy(:,:,opts.apply,:) .* (y(:,:,opts.apply,:) .* (1 - y(:,:,opts.apply,:))) ;
end
