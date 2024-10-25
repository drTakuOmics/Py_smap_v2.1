function outref=dust(volref,criteria,varargin);
% % function outref=dust(volref,criteria,varargin);
% % volref: input volume
% % criteria: 2-element vector: [threshold_value dust_size]
% %     -threshold_value: minimum value (in SDs) to include in the binarized volume
% %     -dust_size: minimum bounding box size of a contiguous region that does not
% %         count as dust. Uses longest edge for this
dust_crit=30; thr_crit=4;

thr_crit=criteria(1);
dust_crit=criteria(2);
cp=floor(size(volref,1)./2)+1;
volref=(volref-nanmean(volref(:)))./nanstd(volref(:));

BW=volref>thr_crit;

s=regionprops3(BW,'Centroid','Volume','BoundingBox','VoxelIdxList');
s.bb_vol=max(s.BoundingBox(:,4:end),[],2);
% s.radius=sqrt(sum((s.Centroid-repmat(cp.*[1 1 1],size(s,1),1)).^2,2));

crit=find(s.bb_vol<dust_crit);
s_eliminate=s(crit,:);
outref=volref;
replace_val=mean(volref(:))-2.*std(volref(:));%mean(volref(:));
dummy=zeros(size(volref));
for i=1:size(s_eliminate,1)
    these_voxels=s_eliminate(i,:).VoxelIdxList{1};
    %         volref_test(these_voxels)=replace_val;
    dummy(these_voxels)=1;
end;
[masked,mask]=smap.mask_a_volume(dummy,[1 3],'mask');
%     [masked,mask]=mask_a_volume(dummy,[1 5],'mask');
outref=outref.*(1-mask);

