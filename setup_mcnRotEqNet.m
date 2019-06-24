function setup_mcnRotEqNet
%SETUP_MCNROTEQNET Sets up mcnExtraLayers by adding its folders to the path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/utils']) ;
