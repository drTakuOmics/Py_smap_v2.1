function [ccOut,peaks]=ccff(imref,templateref,varargin);
% ccOut=ccff(imref_F,template,'filt | noFilt');
ccMode='filt';

imref=(imref-mean(imref(:)))./std(imref(:));

[fPSD,imFilt,~]=smap.psdFilter(imref,'sqrt');       

fullX=size(imref,1);
nTemplates=size(templateref,3);

imref_F=fftn(ifftshift(imFilt));
imrefVec=imref_F(:); %wait(gdev); % normalize
v=sum(abs(imrefVec).^2,1,'native'); %wait(gdev);
v=v./(fullX.*fullX); %wait(gdev);
denom=sqrt(v./(fullX*fullX));
imref_F=imref_F./denom; %wait(gdev);

fPSD=ifftshift(fPSD);

ccOut=zeros(size(imref,1),size(imref,2),nTemplates); 
%ccOut=ones(size(imref,1),size(imref,2)).*-1000;
peaks=[];

%pause

for i=1:nTemplates
    temp=templateref(:,:,i);
    %temp=(temp-mean(temp(:)))./std(temp(:));
    %
    
    switch ccMode
        case 'filt' 
            template=single(smap.extendj(temp,[fullX,fullX],median(temp(:))));
            template=smap.nm(template);
        case 'ew'
        template=single(smap.extendj(temp,[384,384],median(temp(:))));
        template=extraWhiten(template,ntt);
        template=single(smap.extendj(template,[fullX,fullX],median(template(:))));
        otherwise
            template=single(smap.extendj(temp,[fullX,fullX],median(temp(:))));
    end;
    
    
    template_F=fftn(ifftshift(template));%-mean(template(:))));
    switch ccMode
        case 'filt'
            template_F=template_F.*fPSD;
            
%             template_F=fft(template-mean(template(:)),[],2);
%             template=real(ifft(bsxfun(@times,template_F,fPSD),[],2));
%             template_F=fftn(ifftshift(template));%-mean(template(:))));

        otherwise
    end;
    
    templateVec=template_F(:); %wait(gdev); % normalize
    v=sum(abs(templateVec).^2,1,'native'); %wait(gdev);
    v=v./(fullX.*fullX); %wait(gdev);
    denom=sqrt(v./(fullX*fullX));
    template_F=template_F./denom; %wait(gdev);
%    template_F=template_F./0.0033;
    
%     templateVec=template_F(:); %wait(gdev); % normalize
%     v=sum(abs(templateVec).^2,1,'native'); %wait(gdev);
%     v=v./(fullX.*fullX); %wait(gdev);
%     denom=sqrt(v./(fullX*fullX));
%     template_F=template_F./denom; %wait(gdev);

        
%    template_F=template_F./(sqrt(nanvar(template_F(:))./(fullX.*fullX)));
    % convolve the image with the reversed template:
    cc_F=imref_F.*conj(template_F);
    tempCC=real(fftshift(ifftn(cc_F)))./(fullX);
%    figure(3); clf; imsc(tempCC,[1560,2805],200); pause;
    ccOut=tempCC;
    %pause;
    
    %inds=tempCC>ccOut;
    %ccOut(inds)=tempCC(inds);
    peaks(i)=max(tempCC(:));
    if( mod(i,100)==0 )
        fprintf('%d...',i);
    end;
end;
%fprintf('\n');


% %
%function [cc,peaks]=ccff_gpu(imref,templateref,varargin);
% fileNums=1:size(templateref,3);
%         
% nWorkers=8;
% if( length(fileNums)<nWorkers )
%     nWorkers=length(fileNums)
% end;
% nGpus=1;
% gpuIDs=[1];% 2 3 4]
% 
% 
% targetProfileInd=[];
% if( nWorkers<=8 )
%     tempProfiles=parallel.clusterProfiles;
%     for j=1:length(tempProfiles)
%         if( findstr(tempProfiles{j},'profile_8')==1 )
%             targetProfileInd=j;
%             break
%         end;
%     end;
%     if( isempty(targetProfileInd)==1 )
%         myProfile=parallel.importProfile('~/Documents/MATLAB/file_profile_8');
%         parallel.defaultClusterProfile('profile_8');
%     else
%         parallel.defaultClusterProfile(tempProfiles{targetProfileInd});
%     end;
% elseif( nWorkers>8 )
%     tempProfiles=parallel.clusterProfiles;
%     for j=1:length(tempProfiles)
%         if( findstr(tempProfiles{j},'profile_16')==1 )
%             targetProfileInd=j;
%             break
%         end;
%     end;
%     if( isempty(targetProfileInd)==1 )
%         myProfile=parallel.importProfile('~/Documents/MATLAB/file_profile_16');
%         parallel.defaultClusterProfile('profile_16');
%     else
%         parallel.defaultClusterProfile(tempProfiles{targetProfileInd});
%     end;
% end;
% 
% gpuNum=ceil([1:nWorkers]./(nWorkers./nGpus));
% gpuNum=gpuIDs(gpuNum);
% tempvar=matlabpool('size');
% if( tempvar==0 )
%     matlabpool('local',nWorkers);
% end;
% 
% jobsPerWorker=fileNums(end)./nWorkers
% 
% parfor gCounter=1:nWorkers
%     indsToRun{gCounter}=gCounter:nWorkers:fileNums(end)
% %     disp(indsToRun{gCounter});
%     disp(['Running on GPU #' num2str(gpuNum(gCounter))]);
%     [cc_temp{gCounter},peaks_temp{gCounter}]=smap.ccff(imref,templateref(:,:,indsToRun{gCounter}));
% end
% 
% 
% %matlabpool close
% 
% fprintf('combining xcorrs...');
% edgeSize=size(cc_temp{1},1);
% cc=zeros(edgeSize,edgeSize,length(fileNums),'single');;
% for gCounter=1:nWorkers;
%     for i=1:length(indsToRun{gCounter})%jobsPerWorker
%         cc(:,:,indsToRun{gCounter}(i))=cc_temp{gCounter}(:,:,i);
%         peaks(indsToRun{gCounter}(i))=peaks_temp{gCounter}(i);
%     end;
% end;
% fprintf('done.\n');
