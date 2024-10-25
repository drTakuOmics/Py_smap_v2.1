function outref=cropPatchFromImage3(inref,halfDim,rowColInds);
% crop 
%
% function outref=cropPatchFromImage3(inref,halfDim,rowColInds);
    

centerPixel(1)=floor((size(inref,1)/2)+1);
centerPixel(2)=floor((size(inref,2)/2)+1);
shifts=fliplr(centerPixel)-fliplr(rowColInds);
movref_shifted=smap.applyPhaseShifts(inref,shifts);
outref=smap.cutj(movref_shifted,[halfDim.*2 halfDim.*2]);
