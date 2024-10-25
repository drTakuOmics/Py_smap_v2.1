function axan=smap2pymol(RMref,varargin);

%
%q_offset=normalizeRM(RotationMatrix(quaternion.eulerangles('xyz',[180 0 0].*pi./180)));
%q_offset_2=normalizeRM(RotationMatrix(quaternion.eulerangles('xyz',[0 0 -90].*pi./180)));

axan=zeros(size(RMref,3),4);
for i=1:size(RMref,3)

    % % x, y, and z work, but only if final template image is mirrored across x. Solution:
    % in pymol, rotate the object initially upon loading by 180 deg about x; for any subsequent rotations,
    % first rotate back (-180 deg about x), perform rotation, and then rotate forward again (180 x) for display
    %
   theQ=RMref(:,:,i);
    [an,ax]=AngleAxis(quaternion.rotationmatrix(theQ));
    an=an.*180./pi;
    axan(i,:)=[ax' an];
end;
