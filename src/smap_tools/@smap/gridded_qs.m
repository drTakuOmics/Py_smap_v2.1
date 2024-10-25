function q_ref=gridded_qs(range,inc,varargin);

%%

baseVec=[-range:inc:range].*pi./180;
baseVec=baseVec-mean(baseVec);

[xd,yd,zd]=meshgrid(baseVec);
xdd=xd(:); ydd=yd(:); zdd=zd(:);
xyz=[xdd ydd zdd];
R_bump=[];
for i=1:size(xyz,1)
    R_bump(:,:,i)=rotationVectorToMatrix(xyz(i,:));
end;
R_bump=normalizeRM(R_bump);
nBump=size(R_bump,3);
R_bump=normalizeRM(R_bump);

qd=smap.measureQD(eye(3),R_bump);
inds=find(qd<=range);
R_bump=R_bump(:,:,inds);


% clf;
% pr(R_bump,'o');

q_ref=R_bump;


