function outref=cropOrPad(inref,varargin);
% crop or pad an image
% % 
% % cropping: [row column] is optional
% % outref = cropOrPad(inref, [newDims], [row column])
% %
% % padding:
% % outref = cropOrPad(inref, [newDims], [padValue]);
% % 
% %
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

if( sum(abs(oldDim-newDim))==0 )
    outref=inref;
    return;
end;

% %%

cropOrPadFlag=0;
if( min(newDim-oldDim)<0 )
    cropOrPadFlag=1;
elseif( max(newDim-oldDim)>0 )
    cropOrPadFlag=2;
else
    fprintf('crop/pad instructions are unclear...\n');
    return
end;

switch cropOrPadFlag
    case 1 % crop
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
    case 2 % pad
        try
            assert(max(size(varargin{end}))==1);
            padVal=varargin{end};
        catch
            fprintf('no padding value specified; using zero...\n');
            padVal=0;
        end;
        
        % %%
        edges=[];
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
                centerPixNew(i)=floor(halfNewDim(i))+1;
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
            outref=circshift(outref,[centerPixNew(1)-centerPixOld(1),centerPixNew(2)-centerPixOld(2),centerPixNew(3)-centerPixOld(3)]);
        end;
                
end;


