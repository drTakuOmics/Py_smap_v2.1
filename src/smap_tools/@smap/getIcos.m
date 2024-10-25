function [qOut,xyz_sub,xyz_rnap]=getIcos(qBest,varargin);
% flag 0 returns the input q

if( nargin<2 )
    newIcosFlag=1;
else
    newIcosFlag=varargin{1};
end;

if( nargin<3 )
    aPerVox=0.97;
else
    aPerVox=varargin{2};
end;


% this gives identical output for orientations, and locations that differ
% by 1 part in 10,000 (0.0241/147.26 pixels)
load([smap.checkBaseDir 'rotations/qS_PDB.mat'],'OP','qOP','OP_s','qOP_s');
% load('~/matching/rotations/qS.mat','qS');
% load('~/matching/rotaT/mds.mat','mds');
% load('~/matching/rotations/VP6coords.mat','qVP6','vp6_pos');

% % get the set of orientations:
qOut=qBest(1)*qOP_s;
% % get the pre-calculated and DLP-centered COMs for the 60 ASUs:
[xyz_sub,xyz_rnap]=smap.icos(aPerVox);
% % rotate the COMs into alignment with the particle:
xyz_sub=RotateVector(qOut(1),xyz_sub);
xyz_rnap=RotateVector(qOut(1),xyz_rnap);

% xyz_vp6=RotateVector(qOut(1),vp6_pos./aPerVox);
% % and introduce the 180 degree flip:
% qOut=qOut(mds);
