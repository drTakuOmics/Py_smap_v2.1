function outref=cutj(inref,varargin);
% crop an image
% % 072319: added padding function for crop - no longer wraps around when patch size
% % crosses over the image edge. Now instead pads with the inref mean.
% % 072319: added subpixel shifts. Can now center crops on non-integer locations.
% %
% % 062220: combining cutj and extendj into cropOrPad


nDims=length(size(inref));
for i=1:nDims
    oldDim(i)=size(inref,i);
    newDim(i)=varargin{1}(i);
    dimL(i)=(newDim(i)>oldDim(i));
end;

targetXY=[];
if( nargin>2 )
    targetXY=varargin{2};
    cp=floor(oldDim./2)+1;
    margin_size=floor(newDim./2)+1;
    xy_temp=abs(targetXY-cp);
    q1=(cp(1)-margin_size(1));
    q2=(cp(2)-margin_size(2));
    if( (xy_temp(1)>q1) | (xy_temp(2)>q2) )
        spillVal=max([xy_temp(1)-q1 xy_temp(2)-q2]);
        paddedDim=ceil(oldDim+2*spillVal);
        inref=smap.extendj(inref,paddedDim.*ones(size(dimL)),mean(inref(:)));
    end;
    inref=smap.applyPhaseShifts(inref,fliplr(cp-targetXY));
end;

% centerPixOld=smap.getcp(inref);
centerPixOld=floor(size(inref)./2)+1;
for i=1:nDims
    if( dimL(i)==0 )
        oddOldFlag=mod(oldDim(i),2);
        if( oddOldFlag==1 )
            halfOldDim(i)=ceil(oldDim(i)./2);
        else
            halfOldDim(i)=oldDim(i)./2;
        end;
        edges{i}=ceil([centerPixOld(i)-newDim(i)./2 centerPixOld(i)+(newDim(i)./2)-1]);
    else
        edges{i}=[1 oldDim(i)];
    end;
end;

if( nDims<3 )
    outIm=inref(edges{1}(1):edges{1}(2),edges{2}(1):edges{2}(2));
else
    outIm=inref(edges{1}(1):edges{1}(2),edges{2}(1):edges{2}(2),1);
    outIm=zeros(size(outIm,1),size(outIm,2),size(inref,3),class(inref));
    for j=1:oldDim(3)
        outIm(:,:,j)=inref(edges{1}(1):edges{1}(2),edges{2}(1):edges{2}(2),j);
    end;
    outIm=outIm(:,:,(edges{3}(1):edges{3}(2)));
end;
outref=outIm;