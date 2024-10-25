function [filterOut,imOut,psbg]=psdFilter(nfIm,varargin);
% [filterOut,imOut,psbg]=psdFilter(nfIm,varargin);
nSectors=[];
method='psd';
if( nargin>1 )
    method=varargin{1};
    if( nargin>2 )
        nSectors=varargin{2};
    end;
end;

nfIm=nfIm-mean(nfIm(:));
Npix_orig=size(nfIm);
nfIm=smap.cropOrPad(nfIm,max(Npix_orig).*[1,1],0);

switch method
    case 'psd'
        psd=smap.getPSD(nfIm);
    case 'sqrt'
        psd=sqrt(smap.getPSD(nfIm));
end;

if( ~isempty(nSectors) )

edgeSize_blend=21;
% psd_center=smap.radialmeanIm(psd);
[~,psd_center]=smap.radialmeanj(psd,1);
mask=smap.rrj(psd).*size(psd,1);
mask=(mask<=(edgeSize_blend./2));
inds=find(mask==1);

[~,psbg]=smap.radialmeanj(abs(smap.ftj(nfIm)),nSectors);

sf=sum(psbg(inds))./sum(psd_center(inds));
psbg(inds)=((psd_center(inds).*sf)+psbg(inds))./2;

cp=smap.getcp(nfIm);
psbg(cp(1),cp(2))=1;
filterOut=1./psbg;
filterOut(cp(1),cp(2))=0;

imOut=smap.nm(smap.applyFilter(nfIm,filterOut));

else
    
    %pause;
    Npix_im=size(nfIm);
    cp_im=floor(Npix_im./2)+1;
    fAmp=(abs(fftshift(fftn(ifftshift(nfIm))))./sqrt(prod(Npix_im)));    
    [~,fAmp_r_2d]=smap.radialmeanj(fAmp);    
    fAmp_r_2d(cp_im(1),cp_im(2))=1;
    fAmp_r_inv=1./fAmp_r_2d;
    fAmp_r_inv(cp_im(1),cp_im(2))=0;
    fPSD=fAmp_r_inv;

    psbg=fAmp_r_2d;
    filterOut=fPSD;
    imOut=smap.iftj(smap.ftj(nfIm).*fPSD);
    %fAmp_filt=abs(smap.ftj(nfIm_filt));
    
    
    
%     psbg=smap.radialmeanIm(psd);
%     cp=smap.getcp(nfIm);
%     psbg(cp(1),cp(2))=1;
%     filterOut=1./psbg;
%     %filterOut(cp(1),cp(2))=0;
%     filterOut(cp(1),cp(2))=nan;
%     filterOut(cp(1),cp(2))=nanmean(nanmean(filterOut(cp-1:cp+1,cp-1:cp+1)));

    %imOut=smap.nm(smap.applyFilter(nfIm,filterOut));

end;

nfIm=smap.cropOrPad(nfIm,Npix_orig);
imOut=smap.cropOrPad(imOut,Npix_orig);
filterOut=smap.cropOrPad(filterOut,Npix_orig);
psbg=smap.cropOrPad(psbg,Npix_orig);

% % before introducing sectors:
% switch method
%     case 'psd'
%         psd=smap.getPSD(nfIm);
%     case 'sqrt'
%         psd=sqrt(smap.getPSD(nfIm));
% end;
% 
% 
% psbg=smap.radialmeanIm(psd);
% % psbg=smap_radialUnaverage(psd);
% cp=smap.getcp(nfIm);
% psbg(cp(1),cp(2))=1;
% filterOut=1./psbg;
% filterOut(cp(1),cp(2))=0;
% imOut=smap.nm(smap.applyFilter(nfIm,filterOut));

