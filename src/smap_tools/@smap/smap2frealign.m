function [ori,qOut]=smap2frealign(r);
% ori is an Nx3 list of Euler triplets (in degrees)
%load('~/matching/rotations/qS.mat','qS');

qOut=squeeze(quaternion.rotationmatrix(r));
qr=quaternion.eulerangles('xyz',[pi 0 0]);
qOut=qr*qOut;
for i=1:size(r,3)
    oo=(EulerAngles('zyz',qOut(i)))';
    ori(i,:)=fliplr(oo.*180./pi);
    
    
%     % fliplr and convert to radians:
%     oo=fliplr(ori(i,:)).*pi./180; 
%     % convert to quaternion using zyz convention:
%     q=quaternion.eulerangles('zyz',oo);
%     qOut(i)=q; % skipped for phi6
%     r(:,:,i)=RotationMatrix(q);
end;

% for i=1:3
%     inds=find(ori(:,i)<0);
%     ori(inds,i)=360+ori(inds,i);
% end;

ori=real(ori);

% easDeg=deg_f;
% easDegPos=easDeg;
% for m=1:size(easDeg,1)
%     for n=1:size(easDeg,2)
%         if( easDegPos(m,n)<0 )
%             easDegPos(m,n)=360+easDegPos(m,n);
%         end;
%     end;
% end;
% 
% deg_f=easDegPos;

