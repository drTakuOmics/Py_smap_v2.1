function outref=q2R(inref,varargin);
% % function outref=q2R(inref,varargin);
% % inref is a 1x4-element double
%%
%inref=randn(1,4);
% orig=RotationMatrix(quaternion(inref))

denom=sqrt(sum(inref.^2));
if( denom>0 )
    q=inref./denom;
else
    q=[1 0 0 0];
end

R=[ q(1)^2 + q(2)^2 - q(3)^2 - q(4)^2 2*(q(2)*q(3) - q(1)*q(4)) 2*(q(2)*q(4) + q(1)*q(3))
    2*(q(2)*q(3) + q(1)*q(4)) q(1)^2 - q(2)^2 + q(3)^2 - q(4)^2 2*(q(3)*q(4) - q(1)*q(2))
    2*(q(2)*q(4) - q(1)*q(3)) 2*(q(3)*q(4) + q(1)*q(2)) q(1)^2 - q(2)^2 - q(3)^2 + q(4)^2 ];

[U S V]=svd(R);
Rcorrected = U*V'; % Drop the diagonal

outref=Rcorrected;


%%

