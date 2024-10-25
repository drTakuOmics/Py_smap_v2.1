function angles_out=smap2cistem(R_ref);


nAngles=size(R_ref,3);
angles_out=zeros(nAngles,3);
R_ref=normalizeRM(R_ref);
for i=1:nAngles
    R_ref(:,:,i)=R_ref(:,:,i)';
end;
q_ref=squeeze(quaternion.rotationmatrix(R_ref));

angles_out=fliplr(EulerAngles('zyz',q_ref)').*180./pi;


% for i=1:3
%     inds=find(angles_out(:,i)<0);
%     if( length(inds)>0 )
%         angles_out(inds,i)=360+angles_out(inds,i);
%     end;
% end;

angles_out=real(angles_out);


