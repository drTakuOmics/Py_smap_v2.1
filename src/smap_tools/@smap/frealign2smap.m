function [r,qOut]=frealign2smap(ori);
% ori is an Nx3 list of Euler triplets (in degrees)
%load('~/matching/rotations/qS.mat','qS');

r=zeros(3,3,size(ori,1));
qOut=quaternion.zeros(1,size(ori,1));

qr=quaternion.eulerangles('xyz',[pi 0 0]);
    
% qInit=quaternion.eulerangles('zyz',ori(1,:).*pi./180);
for i=1:size(ori,1)
    % fliplr and convert to radians:
    oo=fliplr(ori(i,:)).*pi./180;
    % convert to quaternion using zyz convention:
    q=quaternion.eulerangles('zyz',oo);

    q=qr*q;

    qOut(i)=q;
    r(:,:,i)=RotationMatrix(q);
end;
