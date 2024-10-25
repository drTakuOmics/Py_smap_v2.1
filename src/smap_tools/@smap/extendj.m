function outref=extendj(inref,varargin);
% % pad an image

nDims=length(size(inref));
for i=1:nDims
    oldDim(i)=size(inref,i);
    newDim(i)=varargin{1}(i);
end;
padVal=varargin{end};

for i=1:nDims
    oddOldFlag=mod(oldDim(i),2);
    if( oddOldFlag==1 )
        halfOldDim(i)=ceil(oldDim(i)./2);
        halfNewDim(i)=floor(newDim(i)./2);
        centerPixOld(i)=halfOldDim(i);
        centerPixNew(i)=halfNewDim(i)+1;
    else
        halfOldDim(i)=oldDim(i)./2;
        halfNewDim(i)=newDim(i)./2;
        centerPixOld(i)=halfOldDim(i)+1;
        centerPixNew(i)=floor(halfNewDim(i)+1);
    end;
    edges{i}=[centerPixOld(i)-newDim(i)./2 centerPixOld(i)+(newDim(i)./2)-1];
end;
diffSize=newDim(1)-oldDim(1);

if( nDims<3 )
    rowColInds={ [num2str(centerPixNew(1)-oldDim(1)./2) ':' num2str((centerPixNew(1)+oldDim(1)./2)-1)] , ...
        [num2str(centerPixNew(2)-oldDim(2)./2) ':' num2str((centerPixNew(2)+oldDim(2)./2)-1)] };
    dummy=reshape(1:(newDim(1))*(newDim(2)),(newDim(1)),(newDim(2)));
    eval(['rowColNums=dummy(' rowColInds{1} ',' rowColInds{2} ');']);
    
    if( isa(inref,'gpuArray') )
        outref=gpuArray.ones(newDim(1),newDim(2)).*padVal;
    else
        outref=single(ones(newDim(1),newDim(2)).*padVal);
    end;
    outref(rowColNums)=inref;
    
else
    outref=ones(newDim(1),newDim(2),newDim(3),'single').*padVal;
    for j=1:oldDim(3)
        temp=ones(newDim(1),newDim(2),'single').*padVal;
        temp(1:oldDim(1),1:oldDim(2))=inref(:,:,j);
        outref(:,:,j)=temp;
    end;
    %        outref=circshift(outref,[centerPixNew(1)-centerPixOld(1),centerPixNew(2)-centerPixOld(2),centerPixNew(3)-centerPixOld(3)]);
    outref=circshift(outref,[centerPixNew(1)-centerPixOld(1),centerPixNew(2)-centerPixOld(2),centerPixNew(3)-centerPixOld(3)]);
end;

% end;
