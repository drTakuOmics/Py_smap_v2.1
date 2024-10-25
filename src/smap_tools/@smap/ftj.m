function outref=ftj(inref);
% forward FFT
Npix=prod(size(inref));
inref_F=fftshift(fftn(ifftshift(inref)))./sqrt(Npix);
outref=inref_F;
