function outref=projView(volref);
% make three orthogonal projections from a volume

edge=max(size(volref));
volref=smap.extendj(volref,[edge,edge,edge],median(volref(:)));
outref=zeros(edge,edge,3);
for j=1:3
    outref(:,:,j)=squeeze(sum(volref,j));
end;

