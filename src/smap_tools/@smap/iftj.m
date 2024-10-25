function outref=iftj(inref);
% inverse FFT
Npix=prod(size(inref));
inref(find(isnan(inref)==1))=0;
outref=fftshift(real(ifftn(ifftshift(inref)))).*sqrt(Npix);
