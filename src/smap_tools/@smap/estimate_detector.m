function [filt_out]=estimate_detector(filestruct,varargin);
%
%

%A=dir(fullfile([dirref '/*.mrc'])); %'~/data/060817/mag/060817_*.mrc');

nFrames=length(filestruct);
sample=smap.mr(filestruct(1).name,1,1);
edgeSize=min([size(sample,1) size(sample,2)]);%3456;
cp=floor(edgeSize./2)+1;
%pause;

frames=zeros(edgeSize,edgeSize,nFrames,'single');
ctr=0;
frameToUse=1;
for i=1:nFrames%length(A)
    disp(i);
    try
        temp=smap.mr([filestruct(i).name],frameToUse,1);
        frames(:,:,i)=smap.cutj(temp,[edgeSize,edgeSize]);
        ctr=ctr+1;
    catch
        
    end;
end;
if( ctr<nFrames )
    nFrames=ctr
    frames=frames(:,:,1:nFrames);
end;

im_F_sum=gpuArray.zeros(edgeSize,edgeSize,'single');

ctr=0;
for i=1:nFrames
    a_F=smap.ftj(gpuArray(frames(:,:,i)));
    a_F(cp,cp)=0;
    for j=(i+1):nFrames
        b_F=smap.ftj(gpuArray(frames(:,:,j)));
        b_F(cp,cp)=0;
        im_F_sum=im_F_sum+sqrt(abs(a_F.*conj(b_F))); % x-spectrum
        %im_F_sum=im_F_sum+(abs(a_F.*conj(b_F))); % x-spectrum
        ctr=ctr+1;
    end;
    disp(i);
end;

pause;

mask_floor=smap.rrj(ones(edgeSize,edgeSize)).*edgeSize;
inds=find(abs(mask_floor-(edgeSize/2))<(25));
mask_floor=nan(edgeSize,edgeSize);
mask_floor(inds)=1;

mask_cross=ones(edgeSize,edgeSize);
mask_cross(cp,:)=nan;
mask_cross(:,cp)=nan;
mask_center=smap.rrj(ones(edgeSize,edgeSize)).*edgeSize;
mask_center(mask_center<=3.5)=nan;
mask_center(mask_center>3.5)=1;
mask=mask_center.*mask_cross;

CSD=gather(im_F_sum);
CSD_masked=CSD.*mask;
CSD_floor=nanmean(CSD_masked(:).*mask_floor(:))
CSD_masked(isnan(CSD_masked(:))==1)=CSD_floor;%nanmean(CSD_masked(:));
CSD_masked=CSD_masked./CSD_floor;%./CSD_dcVal;%(sqrt(ctr.*nFrames));%.*1.47);



% CSD_masked=CSD.*mask_cross.*mask_center;
% vals_v=nanmean(CSD_masked(:,(cp-1:cp+1)),2);
% vals_h=nanmean(CSD_masked((cp-1:cp+1),:),1);
% 
% CSD_masked(:,cp)=vals_v;
% CSD_masked(cp,:)=vals_h;
% CSD_masked(cp,cp)=nanmean(nanmean(CSD_masked(cp-1:cp+1,cp-1:cp+1)));
% CSD_masked(isnan(CSD_masked(:)==1))=0;
%pause;

CSD_r=smap.radialmeanIm(CSD_masked);
CSD_r_inv=1./CSD_r;
CSD_r_inv(isnan(CSD_r_inv(:))==1)=1;

filt_out=CSD_r_inv;

%test=nfIm_F./(CSD./sqrt(ctr.*nFrames.*1.47));
