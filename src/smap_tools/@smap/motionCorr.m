%% motion-correction:
function obj=motionCorr(obj,varargin);

finalDownsampling=obj.proc.mcDSFinal;
aPerPix=obj.acq.nmPerPixel_init.*10;
maxShift_A=obj.proc.maxShiftNm.*10;
binVal=obj.proc.mcDSInit;

fprintf('Getting GPU...\n');
gdev=gpuDevice(1);

warning off;

% % hard-coded:
% basedir='/tier2/denk/'; %obj.baseDir;
basedir=smap.checkBaseDir;

inref_fn=obj.proc.GCFilename;
outref_fn=strrep(inref_fn,'gc.mrc','mc.mrc');
outref_sum_fn=strrep(inref_fn,'gc.mrc','mc_fullSum.mrc');
outref_temp_fn=strrep(inref_fn,'gc.mrc','mc_temp.mrc');
fnOut=strrep(inref_fn,'gc.mrc','shifts.mat');

maxShift_pix=maxShift_A./aPerPix;
maxShift_pix=ceil(maxShift_pix./binVal); % search region

% % read in file, get properties and cut to a square:
fprintf('Reading gain-corrected movie...\n');
[movref,rez]=smap.mr(inref_fn);

nFrames=size(movref,3);

tempFrame=movref(:,:,end);
if( mean(tempFrame(:)).*4 < 0.5 )    
    disp('combining low electron-count frames...');
    newmov=zeros(size(movref,1),size(movref,2),size(movref,3)/2,'single');
    ctr=1;
    for j=1:2:nFrames
        newmov(:,:,ctr)=sum(movref(:,:,j:j+1),3);
        ctr=ctr+1;
    end;
    nFrames=nFrames/2;
    obj.proc.exposurePerFrame=obj.proc.exposurePerFrame*2;
    obj.proc.nFrames=obj.proc.nFrames/2;
    movref=newmov; clear newmov;
end;

movDS=smap.resize_F(movref(:,:,1),1./binVal,'newSize');
dsSize=size(movDS,1);
movDS_F=gpuArray(zeros(dsSize,dsSize,nFrames,'single'));
testP=zeros(1,nFrames-1); Y=[]; temp=[];
searchFrames=[10 nFrames-10];

movDS=gpuArray(zeros(dsSize,dsSize,nFrames,'single'));

if( binVal>1 )
    fprintf('Resampling...\n');
    fprintf('Finding reference frame...\n');
    firstFrame=gpuArray(smap.nm(movref(:,:,1)));
    firstFrame_F=smap.cutj(smap.ftj(firstFrame),[dsSize,dsSize]);
    movDS_F(:,:,1)=firstFrame_F;
    movDS(:,:,1)=smap.iftj(movDS_F(:,:,1));
    for j=2:nFrames
        disp(j);
        nextFrame=gpuArray(smap.nm(movref(:,:,j)));
        nextFrame_F=smap.cutj(smap.ftj(nextFrame),[dsSize,dsSize]);
        tempCC=smap.cutj(real(fftshift(ifftn(ifftshift(firstFrame_F.*conj(nextFrame_F))))),[maxShift_pix*2,maxShift_pix*2]); wait(gdev);
        peakVal=max(tempCC(:)); wait(gdev);
        testP(j-1)=gather(peakVal); wait(gdev);
        movDS_F(:,:,j)=nextFrame_F;
        movDS(:,:,j)=smap.iftj(movDS_F(:,:,j));
        firstFrame_F=nextFrame_F;
    end;
    
%     movref=single(temp);
end;
clear nextFrame nextFrame_F;

testP(setdiff(1:nFrames,searchFrames(1)+1:searchFrames(2)))=0;
refNum=find(testP==max(testP),1,'first')

doneFrames=zeros(1,nFrames);
shifts=zeros(nFrames,2);
refImOrig=movref(:,:,refNum);
peaks=gpuArray.zeros(1,nFrames);
peaksBest=peaks;
nDone=1;

fprintf('Starting motion-correction...\n');
ctr=1; pbd=gpuArray.zeros(1);
shiftsAll=gpuArray.zeros(nFrames,2);
pbd=gpuArray([1e7 1e8]);
shiftsByRound=[];
annealingMode=0;
doneFrames(refNum)=1;
goodFrames=gpuArray(find(doneFrames==1));
movNew=gpuArray(zeros(dsSize,dsSize,nFrames,'single'));
movNew(:,:,refNum)=movDS(:,:,refNum);
while( ((ctr<100)&(abs(gather(pbd(end))-gather(pbd(1:end-1)))>0))|(ctr<=nFrames) )
    if( ctr>=nFrames )
        annealingMode=1;
    end;
    
    % % self-excluding stack of summed reference images:
    shifts=gpuArray.zeros(nFrames,2);
    peakVals=gpuArray.zeros(1,nFrames);

    for j=1:nFrames
        if( (j==refNum) && (ctr==1) )
            
        else
            refIm_F=smap.ftj(smap.nm(sum(movNew(:,:,setdiff(1:nFrames,j)),3)));
            tempCC=real(fftshift(ifftn(ifftshift(refIm_F.*conj(movDS_F(:,:,j))))));
            tempCC=smap.cutj(tempCC,[maxShift_pix,maxShift_pix]);
            [shifts(j,:),peakVals(j)]=smap.maxInterpF(tempCC,maxShift_pix,10*binVal);
        end;
    end;

    
    % apply and store:
    peaksNew=peakVals;
    
    pb=[peaksBest; peaksNew];
    peaksBest=max(peaksBest,peaksNew);
    improvedPeaks=find(pb(2,:)>pb(1,:));
    pbd(ctr)=sum(abs(pb(2,:)-pb(1,:)));
    if( nDone==1 )
        [X,Y]=sort(peaksNew,'descend');
        newFrames=Y(1:2); % ref frame and next-best frame
        shiftsAll(newFrames,:)=shifts(newFrames,:);%.*binVal;
    else
        peaksEx=(1-doneFrames).*peaksNew;
        newFrames=find(peaksEx==max(peaksEx));
        dpbd=[0 diff(pbd)];
    end;
    if( ~isempty( newFrames ) )
        doneFrames(newFrames)=1;
        goodFrames=gpuArray(find(doneFrames==1)); % why is this necessary?
        shiftsAll(newFrames,:)=shifts(newFrames,:);%.*binVal;
        shiftsToUpdate=intersect(gather(improvedPeaks),gather(goodFrames));
        if( isempty(shiftsToUpdate) )
            shiftsToUpdate=newFrames;
        end;
        shiftsAll(shiftsToUpdate,:)=shifts(shiftsToUpdate,:);
        
        if( ~isempty(goodFrames) )%(shiftsToUpdate) )            
%             for j=1:length(goodFrames)%(shiftsToUpdate)
%                 movNew(:,:,goodFrames(j))=smap.iftj(smap.applyPhaseShifts(movDS_F(:,:,goodFrames(j)),-fliplr(shiftsAll(goodFrames(j),:))));
% %                 movNew(:,:,shiftsToUpdate(j))=smap.iftj(smap.applyPhaseShifts(movDS_F(:,:,shiftsToUpdate(j)),-fliplr(shiftsAll(shiftsToUpdate(j),:)))); wait(gdev);
%             end;
%         end;
%         movNew_F=smap.applyPhaseShifts(movDS_F,-fliplr(shiftsAll)); wait(gdev);
        movNew=gpuArray(smap.applyPhaseShifts(movDS,-fliplr(shiftsAll))); wait(gdev);
        
        end;
    end;
        
    
    %figure(101); subplot(1,2,1);
    plot(shiftsAll,'x-');
    xlim([-0.5 nFrames+0.5]); ylim([-20 20]); pause(0.05);
    
    shiftsByRound{ctr}=shiftsAll;
    
    nDone=length(find(doneFrames==1));
    fprintf('%d %f\n',[ctr pbd(ctr)]);
    save(fnOut,'shiftsByRound');
    ctr=ctr+1;
    %                 toc;
end;
fprintf('...done.\n');

pos=sqrt(sum(shiftsAll.^2,2));
pd=[0; diff(pos)];
goodFrames=find(pd<2.*std(pd));
badFrames=setdiff(1:size(shiftsAll,1),goodFrames);
if( ~isempty(badFrames) )
    shiftsAll(badFrames,1)=interp1(goodFrames,shiftsAll(goodFrames,1),badFrames,'linear');
    shiftsAll(badFrames,2)=interp1(goodFrames,shiftsAll(goodFrames,2),badFrames,'linear');
end;

shiftsFinal=shiftsAll-repmat(shiftsAll(refNum,:),size(shifts,1),1);
%shiftsFinal=shiftsFinal.*binVal;

temp=smap.resize_F(movref(:,:,1),1./finalDownsampling,'newSize');
dsSize=size(temp,1);
movShifted=gpuArray(zeros(dsSize,dsSize,nFrames,'single'));

clear movDS movNew

fprintf('Applying final shifts...\n');
for j=1:nFrames
    firstFrame=gpuArray(movref(:,:,j));
    firstFrame=smap.applyPhaseShifts(firstFrame,-fliplr(shiftsFinal(j,:).*binVal));
    movShifted(:,:,j)=smap.resize_F(firstFrame,0.5,'newSize');   
    disp(j);
%     firstFrame_F=smap.cutj(fftshift(fftn(ifftshift(firstFrame))),[dsSize,dsSize]);
%     movShifted(:,:,j)=smap.iftj(smap.applyPhaseShifts(firstFrame_F,-fliplr(shiftsFinal(j,:)./(0.5.*finalDownsampling))));
    %movShifted(:,:,j)=smap.iftj(smap.applyPhaseShifts(firstFrame_F,-fliplr(shiftsFinal(j,:)./finalDownsampling)));
end;

movShifted=gather(movShifted);

fullSum=sum(movShifted,3);
fprintf('Writing motion-corrected files...\n');

smap.mw(single(fullSum),outref_sum_fn,aPerPix.*finalDownsampling);
smap.mw(single(movShifted),outref_fn,aPerPix.*finalDownsampling);

obj.proc.motionTracking=shiftsFinal.*aPerPix;
for j=1:length(shiftsByRound)
    shiftsByRound{j}=shiftsByRound{j}.*aPerPix;
end;
% obj.proc.motionTracking.shiftsByRound=shiftsByRound;
obj.proc.fullSum_image=outref_sum_fn;
obj.proc.MCFilename=outref_fn;

% s=obj;
% save(['/tier2/denk/datasets/' obj.prop.ID.ID '.mat'],'s');

obj.final.nmPerPixel=obj.acq.nmPerPixel_init.*obj.proc.mcDSFinal;
obj.CTF.input_filename=obj.proc.fullSum_image;
obj.CTF.output_diag_filename=strrep(obj.CTF.input_filename,'.mrc','CTFFindOutput.mrc');
obj.CTF.pixel_size=obj.final.nmPerPixel*10;

fprintf('Removing gain-corrected stack...');
% delete(inref_fn);
obj.ID.MC=1;
s=obj;
% save(['/tier2/denk/datasets/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'datasets/' obj.ID.ID '.mat'],'s');

disp('done');


%%
% %% motion-correction:
% function obj=motionCorr(obj,varargin);
% 
% finalDownsampling=obj.proc.mcDSFinal;
% aPerPix=obj.acq.nmPerPixel_init.*10;
% maxShift_A=obj.proc.maxShiftNm.*10;
% binVal=obj.proc.mcDSInit;
% 
% fprintf('Getting GPU...\n');
% gdev=gpuDevice(1);
% 
% warning off;
% 
% % % hard-coded:
% basedir='/tier2/denk/'; %obj.baseDir;
% 
% inref_fn=obj.proc.GCFilename;
% outref_fn=strrep(inref_fn,'gc.mrc','mc.mrc');
% outref_sum_fn=strrep(inref_fn,'gc.mrc','mc_fullSum.mrc');
% outref_temp_fn=strrep(inref_fn,'gc.mrc','mc_temp.mrc');
% fnOut=strrep(inref_fn,'gc.mrc','shifts.mat');
% 
% maxShift_pix=maxShift_A./aPerPix;
% maxShift_pix=ceil(maxShift_pix./binVal); % search region
% 
% % % read in file, get properties and cut to a square:
% fprintf('Reading gain-corrected movie...\n');
% [movref,rez]=smap.mr(inref_fn);
% nFrames=size(movref,3);
% temp=smap.resize_F(movref(:,:,1),1./binVal,'newSize');
% temp=zeros(size(temp,1),size(temp,2),size(movref,3),'single');
% dsSize=floor(min([size(movref,1) size(movref,2)])./binVal);
% testP=zeros(1,nFrames-1); Y=[]; temp=[];
% searchFrames=[10 nFrames-10];
% 
% if( binVal>1 )
%     fprintf('Resampling...\n');
%     fprintf('Finding reference frame...\n');
%     firstFrame=gpuArray(movref(:,:,searchFrames(1)));
%     firstFrame_F=smap.cutj(fftshift(fftn(ifftshift(firstFrame))),[dsSize,dsSize]); wait(gdev);    
%     for j=searchFrames(1)+1:searchFrames(2);
%         disp(j);        
%         nextFrame=gpuArray(movref(:,:,j));
%         nextFrame_F=smap.cutj(fftshift(fftn(ifftshift(nextFrame))),[dsSize,dsSize]); wait(gdev);
%         tempCC=smap.cutj(real(fftshift(ifftn(ifftshift(firstFrame_F.*conj(nextFrame_F))))),[maxShift_pix*2,maxShift_pix*2]); wait(gdev);
%         peakVal=max(tempCC(:)); wait(gdev);
%         testP(j-1)=gather(peakVal); wait(gdev);
%         firstFrame_F=nextFrame_F;
%     end;
% %     movref=single(temp);
% end;
% clear nextFrame nextFrame_F;
% 
% % minDim=min([size(movref,1) size(movref,2)]);
% % temp=movref; movref=zeros(minDim,minDim,nFrames,'single');
% % for j=1:nFrames
% %     movref(:,:,j)=smap.nm(smap.cutj(temp(:,:,j),[minDim,minDim]));
% % end;
% 
% % % get initial reference frame (standard: 10-30 for 40 Hz):
% % fprintf('Finding reference frame...\n');
% % searchFrames=[10 size(movref,3)-10];
% % testP=zeros(1,nFrames); Y=[]; temp=[];
% % for j=searchFrames(1):searchFrames(2)
% %     temp=smap.ccf(movref(:,:,j),movref(:,:,j+1));
% %     testP(j)=max(temp(:));
% % end;
% refNum=find(testP==max(testP),1,'first')+1
% 
% doneFrames=zeros(1,nFrames);
% shifts=zeros(nFrames,2);
% refImOrig=movref(:,:,refNum);
% peaks=zeros(1,nFrames);
% peaksBest=peaks;
% nDone=1;
% 
% refIm=refImOrig;
% 
% inputMov=movref;
% movref=gpuArray(movref); wait(gdev);
% movNew=movref;
% 
% fprintf('Starting motion-correction...\n');
% movref_F=gpuArray(complex(zeros(size(movref),'single'),zeros(size(movref),'single')));
% for j=1:nFrames
%     movref_F(:,:,j)=smap.ftj(gpuArray(movref(:,:,j)));
% end;
% ctr=1; pbd=[];
% shiftsAll=zeros(nFrames,2);
% pbd=[1e7 1e8];
% shiftsByRound=[];
% annealingMode=0;
% while( ((ctr<100)&(abs(pbd(end)-pbd(1:end-1))>0))|(ctr<=nFrames) )
%     if( ctr>=nFrames )
%         annealingMode=1;
%     end;
%     
%     for j=1:nFrames
%         refIm_F(:,:,j)=smap.ftj(smap.nm(sum(movNew(:,:,setdiff(1:nFrames,j)),3)));
%     end;
%     
%     shifts=zeros(nFrames,2);
%     peakVals=zeros(1,nFrames);
%     for j=1:nFrames
%         cc=gather(smap.iftj(refIm_F(:,:,j).*conj(movref_F(:,:,j)))); wait(gdev);
%         [shifts(j,:),peakVals(j)]=smap.maxInterpF(cc,maxShift_pix,10*binVal);
%     end;
%     
%     % apply and store:
%     peaksNew=peakVals;
%     
%     pb=[peaksBest; peaksNew];
%     peaksBest=max(peaksBest,peaksNew);
%     improvedPeaks=find(pb(2,:)>pb(1,:));
%     pbd(ctr)=sum(abs(pb(2,:)-pb(1,:)));
%     if( nDone==1 )
%         [X,Y]=sort(peaksNew,'descend');
%         newFrames=Y(1:2); % ref frame and next-best frame
%         shiftsAll(newFrames,:)=shifts(newFrames,:);%.*binVal;
%     else
%         peaksEx=(1-doneFrames).*peaksNew;
%         newFrames=find(peaksEx==max(peaksEx));
%         dpbd=[0 diff(pbd)];
%     end;
%     if( ~isempty( newFrames ) )
%         doneFrames(newFrames)=1;
%         goodFrames=find(doneFrames==1);
%         shiftsAll(newFrames,:)=shifts(newFrames,:);%.*binVal;
%         shiftsToUpdate=intersect(improvedPeaks,goodFrames);
%         shiftsAll(shiftsToUpdate,:)=shifts(shiftsToUpdate,:);
%         %                     if( isempty(shiftsToUpdate)==1 )
%         %                         temp=smap.applyPhaseShifts(movref(:,:,newFrames),-fliplr(shiftsAll(newFrames,:)));
%         %                         movNew(:,:,newFrames)=temp;
%         %                     else
%         %                         temp=smap.applyPhaseShifts(movref(:,:,shiftsToUpdate),-fliplr(shiftsAll(shiftsToUpdate,:)));
%         %                         for j=1:size(shiftsToUpdate,2)
%         %                             movNew(:,:,shiftsToUpdate(j))=temp(:,:,j);
%         %                         end;
%         %                     end;
%         movNew=gpuArray(smap.applyPhaseShifts(inputMov,-fliplr(shiftsAll))); wait(gdev);
%     end;
%     
%     figure(101); subplot(1,2,1);
%     plot(shiftsAll,'x-');
%     xlim([-0.5 nFrames+0.5]); ylim([-maxShift_pix maxShift_pix]); pause(0.05);
%     
%     shiftsByRound{ctr}=shiftsAll;
%     
%     nDone=length(find(doneFrames==1));
%     fprintf('%d %f\n',[nDone pbd(ctr)]);
%     save(fnOut,'shiftsByRound');
%     ctr=ctr+1;
%     %                 toc;
% end;
% fprintf('...done.\n');
% 
% % frameByFrameShifts=max(abs(shifts),[],2);
% % frameByFrameDiff=[0; diff(frameByFrameShifts)];
% % frameByFrameDiff_2=[diff(frameByFrameShifts); 0];
% 
% pos=sqrt(sum(shiftsAll.^2,2));
% pd=[0; diff(pos)];
% goodFrames=find(pd<2.*std(pd));
% badFrames=setdiff(1:size(shiftsAll,1),goodFrames);
% shiftsAll(badFrames,1)=interp1(goodFrames,shiftsAll(goodFrames,1),badFrames,'linear');
% shiftsAll(badFrames,2)=interp1(goodFrames,shiftsAll(goodFrames,2),badFrames,'linear');
% 
% shiftsFinal=shiftsAll-repmat(shiftsAll(refNum,:),size(shifts,1),1)
% 
% movref=smap.mr(inref_fn);
% ftu=1:size(movref,3)
% movShifted=single(smap.applyPhaseShifts(movref(:,:,ftu),-fliplr(shiftsFinal).*binVal));
% 
% minDim=min([size(movShifted,1) size(movShifted,2)]);
% temp=movShifted; movShifted=zeros(minDim,minDim,nFrames,'single');
% for j=1:nFrames
%     movShifted(:,:,j)=smap.cutj(temp(:,:,j),[minDim,minDim]);
% end;
% 
% movShifted=movShifted(:,:,ftu);
% temp=movShifted; clear movShifted;
% for j=1:nFrames
%     disp(j);
%     movShifted(:,:,j)=smap.resize_F(temp(:,:,j),1./finalDownsampling,'newSize');
% end;
% 
% fullSum=sum(movShifted,3);
% fprintf('Writing motion-corrected files...\n');
% smap.mw(single(fullSum),outref_sum_fn,aPerPix.*finalDownsampling);
% smap.mw(single(movShifted),outref_fn,aPerPix.*finalDownsampling);
% 
% % obj.proc.motionTracking.shiftsAll=shiftsAll.*binVal.*aPerPix;
% obj.proc.motionTracking=shiftsFinal.*binVal.*aPerPix;
% for j=1:length(shiftsByRound)
%     shiftsByRound{j}=shiftsByRound{j}.*binVal.*aPerPix;
% end;
% % obj.proc.motionTracking.shiftsByRound=shiftsByRound;
% obj.proc.fullSum_image=outref_sum_fn;
% obj.proc.MCFilename=outref_fn;
% 
% % s=obj;
% % save(['/tier2/denk/datasets/' obj.prop.ID.ID '.mat'],'s');
% 
% obj.final.nmPerPixel=obj.acq.nmPerPixel_init.*obj.proc.mcDSFinal;
% obj.CTF.input_filename=obj.proc.fullSum_image;
% obj.CTF.output_diag_filename=strrep(obj.CTF.input_filename,'.mrc','CTFFindOutput.mrc');
% obj.CTF.pixel_size=obj.final.nmPerPixel*10;
% 
% fprintf('Removing gain-corrected stack...');
% % delete(inref_fn);
% obj.ID.MC=1;
% s=obj;
% save(['/tier2/denk/datasets/' obj.ID.ID '.mat'],'s');
% 
% disp('done');
% 
% 

