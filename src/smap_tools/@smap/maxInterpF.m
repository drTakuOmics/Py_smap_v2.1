function [nxny,peakVal,errorFlags]=maxInterpF(imref,varargin);
% interpolate by Fourier blowing-it-up
% [nxny,peakVal,errorFlags]=maxInterpF(imref,varargin);
% varargin can include: [halfWidth],[interpFactor],[centerPixel_target]

rtu=min(size(imref));
iFactor=4;
cp_init=floor((size(imref,1)/2)+1).*[1 1];
cp_full=cp_init;
cp_flag=0;
if( nargin>1 )
    rtu=varargin{1};
    if( nargin>2 )
        iFactor=varargin{2};
        if( nargin>3 )
            %cp_init=varargin{3};
            
            cp_flag=1;
            
        end;
    end;
end;
cc=smap.cropPatchFromImage3(imref,rtu,[cp_init]);
%cc=imref;
cp=floor((size(cc,1)/2)+1);
if( cp_flag==0 )
    [nx,ny]=find(cc==max(cc(:)),1,'first');
else
    nx=cp; ny=cp;
end;

shifts_fullpix=-([cp cp]-[nx ny]);
ccPatch=smap.cropPatchFromImage3(cc,rtu,[nx ny]);
ccPatch_i=smap.resize_F(ccPatch,iFactor,'newSize');
cp_i=floor((size(ccPatch_i,1)/2)+1);
nanmask=smap.rrj(ccPatch_i);
nanmask(find(nanmask>0.5))=nan;
[nx_i,ny_i]=find(ccPatch_i==nanmax(ccPatch_i(:)),1,'first');
shifts_subpix=-([cp_i cp_i]-[nx_i ny_i])./iFactor;
shifts=shifts_fullpix+shifts_subpix;
shiftsToUse=-fliplr(shifts);
nxny=-shifts;
peakVal=max(ccPatch_i(:));

nxny=(cp_full-cp_init)+nxny;
