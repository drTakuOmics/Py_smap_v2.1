function angles_out=cistem2smap(EA_ref);


nAngles=size(EA_ref,1);
angles_out=zeros(3,3,nAngles);
angles_out=squeeze(RotationMatrix(quaternion.eulerangles('zyz',fliplr(EA_ref.*pi./180))));
for i=1:nAngles
    angles_out(:,:,i)=angles_out(:,:,i)';
end;

angles_out=normalizeRM(angles_out);

