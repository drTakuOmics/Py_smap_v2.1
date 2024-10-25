function psd=getPSD(nfIm,varargin);

nPix=size(nfIm,1);
%nfIm=nfIm-nanmean(nfIm(:));
nfIm_F=abs(fftshift(fftn(ifftshift(nfIm)))).^2;
cp=smap.getcp(nfIm);
nfIm_F(cp(1),cp(2))=0;
psd=((nfIm_F)./(nPix.^2));%.^2;

% % % psd=(abs(fftshift(fft2(ifftshift((nfIm-nanmean(nfIm(:))))))).^2);

%psd=(abs(fftshift(fft2(ifftshift(nfIm))./(nPix.^2))).^2);

