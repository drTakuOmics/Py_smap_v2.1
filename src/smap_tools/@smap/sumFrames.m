function obj=sumFrames(obj,varargin);

targetExposure=obj.proc.targetExposure; % in e/nm^2, e.g., 1200;
nToUse=ceil(targetExposure./obj.proc.exposurePerFrame)
fn=obj.proc.MCFilename;
[temp,rez]=smap.mr(fn);
nToUse=min([nToUse size(temp,3)])

summedMov=sum(temp(:,:,1:nToUse),3);
summedMov=smap.nm(summedMov);

if( (obj.proc.addMask==1) )
    if( isempty(obj.final.mask_image) )
        temph=figure(101); clf;
        imagesc(summedMov); set(gcf,'Position',[0          31        1178        1075]);
        axis equal; colormap(gray); axis off;
        n=input('number of beads? ')
        BW=[];
        if( n>0 )
            for i=1:n
                words=['select polygon for mask ' num2str(i) '...']; title(words);
                BW(:,:,i)=roipoly;
            end;
        else
            BW=zeros(size(inputIm));
        end;
        mask=double(max(BW,[],3));
        maskOut=mask;
        close(temph);
        maskToUse=1-maskOut;
        fprintf('Writing %s...\n',obj.final.mask_image);
        smap.mw(single(maskToUse),obj.final.mask_image,rez);
%         obj.final.mask_image=['/tier2/denk/images/search/' obj.ID.ID '_mask.mrc'];
        obj.final.mask_image=[smap.checkBaseDir 'images/search/' obj.ID.ID '_mask.mrc'];
    else
        maskToUse=ri(obj.final.mask_image);
    end;
    % create an image where beads are replaced with mean val of frame (excluding bead regions):
    
    nanmask=maskToUse;
    nanmask(find(maskToUse==0))=nan;
    temp=summedMov.*nanmask;
    temp=smap.nm(temp);
    %     temp=(temp-nanmean(temp(:)))./nanstd(temp(:));
    temp(find(isnan(temp)))=nanmean(temp(:));
    outIm=smap.nm(temp);
    
else
    outIm=summedMov;
    %     obj.search.maskFilename=['/tier2/denk/images/search/' obj.ID.ID '_mask.mrc'];
    fprintf('no masking...');
end;


% obj.final.search_image=['/tier2/denk/images/search/' obj.ID.ID '_search.mrc'];
obj.final.search_image=[smap.checkBaseDir 'images/search/' obj.ID.ID '_search.mrc'];
fprintf('Writing %s...\n',obj.final.search_image);
smap.mw(single(outIm),obj.final.search_image,rez);

obj.ID.SF=1;

s=obj;
% save(['/tier2/denk/datasets/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'datasets/' obj.ID.ID '.mat'],'s');

disp('done');

