function outref=tileImages(imstack);
% compose a tiled image from a three-page stack (or cell array)

if( iscell(imstack)==0 )
    for j=1:3
        projIm{j}=imstack(:,:,j);
    end;
else
    projIm=imstack;
end;

outref=nan(size(projIm{1},1).*2);
outref(1:size(projIm{1},1),1:size(projIm{1},1))=projIm{1};
outref(1:size(projIm{1},1),(size(projIm{1},1)+1):(2*size(projIm{1},1)))=projIm{2};
outref((size(projIm{1},1)+1):(2*size(projIm{1},1)),1:(size(projIm{1},1)))=projIm{3};
pm=nanmedian(outref(find(isnan(outref)==0)));
outref(isnan(outref))=pm;
