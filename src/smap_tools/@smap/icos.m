function [xyz_sub,xyz_rnap]=icos(varargin);
if( nargin<1 )
    aPerVox=0.97;
else
    aPerVox=varargin{1};
end;

load('~/rotations/rotaXYZ.mat','rotaXYZ');
load('~/rotations/rnapXYZ.mat','rnapXYZ');

COM=mean(rotaXYZ,2);
COMshift=repmat(COM,1,60);
xyzR=rotaXYZ-COMshift;
xyzP=rnapXYZ-COM;

%xyzR(3,:)=-xyzR(3,:);

xyz_sub=xyzR.*(0.97./aPerVox);
xyz_rnap=xyzP.*(0.97./aPerVox);


