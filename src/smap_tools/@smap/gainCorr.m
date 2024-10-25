%% gain-correction:
function obj=gainCorr(obj); 
        
        
% basedir='/tier2/denk/'; %obj.baseDir;
basedir=smap.checkBaseDir;
acqDate=obj.acq.acqDate;

gainref_fn=obj.acq.gainRefFilename;
inref_fn=obj.acq.raw_image;
% outref_fn=['/tier2/denk/images/preproc/' obj.ID.ID '_gc.mrc'];
outref_fn=[smap.checkBaseDir 'images/preproc/' obj.ID.ID '_gc.mrc'];

disp('loading gain ref...');
[m]=smap.ri(gainref_fn);


if( isempty(obj.acq.defocusFilename)==0 )
    disp('loading paired defocused image...');
    tempOrig=smap.ri(obj.acq.defocusFilename);
    outref_fnDf=strrep(outref_fn,'_gc.mrc','_df_ds.mrc');

    pixelSize=obj.acq.nmPerPixel_init;
    shortEdge=min([size(tempOrig,1) size(tempOrig,2)]);
    
    tempOrig=smap.cutj(tempOrig,[shortEdge,shortEdge,size(tempOrig,3)]);
    
    temp=zeros(size(tempOrig),'single');
%     m=smap.cutj(m,[shortEdge,shortEdge]);
    
    gcAll=zeros(size(tempOrig),'single');
    
    for j=1:size(tempOrig,3)
        temp(:,:,j)=tempOrig(:,:,j)';
    end;
    
%     tempDs=[];
%     for j=1:size(temp,3)
%         gcAll(:,:,j)=temp(:,:,j).*m;
%         tempDs(:,:,j)=smap.resize_F(gcAll(:,:,j),0.25,'newSize');
%         tempDiff=abs(diff(tempDs(:,:,j),1)); % diff along dim 2 if im not rotated/flipped
%         nZeros(j)=length(find(tempDiff==0));
%     end;
%     
%     badFrames=find(nZeros>100);
%     if( length(badFrames)>0 )
%         disp(['found ' num2str(length(badFrames)) 'bad frames...']);
%     end;
%     
%     goodFrames=setdiff(1:size(gcAll,3),badFrames)
%     gcAll=gcAll(:,:,goodFrames);
%     temp2=sum(gcAll,3);
    
    gcAll=sum(temp,3);
    clear temp;
    
    gs=sum(gcAll,3);
    [nx_hot,ny_hot]=find(smap.nm(gs)>7);
    nx=nx_hot; ny=ny_hot;
    allPix=[nx ny];
    disp(['replacing hot pixels... (found ' num2str(length(nx_hot)) ')']);    
    
    imOut=zeros(shortEdge,shortEdge,size(gcAll,3),'single');
    for j=1:size(gcAll,3)
        movref=gcAll(:,:,j);
        outIm=movref;
        for i=1:size(allPix,1)
            rs=[(allPix(i,1)-1):(allPix(i,1)+1)];
            cs=[(allPix(i,2)-1):(allPix(i,2)+1)];
            if(min(rs)<1)
                rs=rs-min(rs)+1;
            end;
            if(min(cs)<1)
                cs=cs-min(cs)+1;
            end;
            if( max(rs)>size(movref,1) )
                rs=rs-(max(rs)-size(movref,1));
            end;
            if( max(cs)>size(movref,2) )
                cs=cs-(max(cs)-size(movref,2));
            end;
            tempPatch=movref(rs,cs);
            mv=mean(tempPatch(:));
            outIm(allPix(i,1),allPix(i,2))=mv;
        end;
        disp(j);
        imOut(:,:,j)=outIm;
    end;
    
    imToWrite=sum(imOut,3);
    imToWrite=smap.resize_F(imToWrite,0.25,'newSize');
    disp('Saving paired defocus image...');
    smap.mw(imToWrite,outref_fnDf,pixelSize*10*4);
    
    obj.proc.defocused_image=outref_fnDf;

end;
    



disp('loading raw image file...');
tempOrig=smap.tr(inref_fn);

pixelSize=obj.acq.nmPerPixel_init;
obj.proc.nFrames=size(tempOrig,3);
obj.proc.exposurePerFrame=(sum(tempOrig(:))./(size(tempOrig,1).*size(tempOrig,2).*size(tempOrig,3))).*(1./((pixelSize).^2));

% tempOrig=tempOrig(:,:,2:end);
shortEdge=min([size(tempOrig,1) size(tempOrig,2)]);

tempOrig=smap.cutj(tempOrig,[shortEdge,shortEdge,size(tempOrig,3)]);

temp=zeros(size(tempOrig),'single');
m=smap.cutj(m,[shortEdge,shortEdge]);

gcAll=zeros(size(tempOrig),'single');

for j=1:size(tempOrig,3)
    temp(:,:,j)=tempOrig(:,:,j)';
end;

tempDs=[];
for j=1:size(temp,3)
    gcAll(:,:,j)=temp(:,:,j).*m;
    tempDs(:,:,j)=smap.resize_F(gcAll(:,:,j),0.125,'newSize');
    tempDiff=abs(diff(tempDs(:,:,j),1)); % diff along dim 2 if im not rotated/flipped
    nZeros(j)=length(find(tempDiff==0));
end;

badFrames=find(nZeros>100);
if( length(badFrames)>0 )
    disp(['found ' num2str(length(badFrames)) 'bad frames...']);
end;

obj.proc.badFrames=badFrames;

goodFrames=setdiff(1:size(gcAll,3),badFrames)
gcAll=gcAll(:,:,goodFrames);
temp2=sum(gcAll,3);

clear temp;

gs=sum(gcAll,3);
[nx_hot,ny_hot]=find(smap.nm(gs)>7);
nx=nx_hot; ny=ny_hot;
allPix=[nx ny];
disp(['replacing hot pixels... (found ' num2str(length(nx_hot)) ')']);

obj.proc.hotPixels=allPix;

imOut=zeros(shortEdge,shortEdge,size(gcAll,3),'single');
for j=1:size(gcAll,3)
    movref=gcAll(:,:,j);
    outIm=movref;
    for i=1:size(allPix,1)
        rs=[(allPix(i,1)-1):(allPix(i,1)+1)];
        cs=[(allPix(i,2)-1):(allPix(i,2)+1)];
        if(min(rs)<1)
            rs=rs-min(rs)+1;
        end;
        if(min(cs)<1)
            cs=cs-min(cs)+1;
        end;
        if( max(rs)>size(movref,1) )
            rs=rs-(max(rs)-size(movref,1));
        end;
        if( max(cs)>size(movref,2) )
            cs=cs-(max(cs)-size(movref,2));
        end;
        tempPatch=movref(rs,cs);
        mv=mean(tempPatch(:));
        outIm(allPix(i,1),allPix(i,2))=mv;
    end;
    disp(j);
    imOut(:,:,j)=outIm;
end;

clear gcAll tempOrig;

disp('Saving gain-corrected mov..');
smap.mw(imOut,outref_fn,pixelSize*10);

obj.proc.GCFilename=outref_fn;
obj.ID.GC=1;

s=obj;
% save(['/tier2/denk/datasets/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'datasets/' obj.ID.ID '.mat'],'s');

disp('done');
