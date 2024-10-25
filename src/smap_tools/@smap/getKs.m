function [k_2d,centerPixel]=getKs(imref,aPerPix);
% function [k_2d,centerPixel]=getKs(imref,aPerPix);

if( (size(imref,1)==1) & (size(imref,2)==1) )
    imref=zeros(imref,imref,'single');
end;

Npix=size(imref,1);
centerPixel=floor((Npix./2)+1);
k_2d=smap.rrj(imref)./aPerPix;

% if( (size(imref,1)==1) & (size(imref,2)==1) )
%     imref=zeros(imref,imref,'single');
% end;
% 
% Npix=size(imref,1);
% centerPixel=floor((Npix./2)+1);
% % k_2d=smap_rrj(imref)./aPerPix;
% k_2d=smap.rrj(imref)./aPerPix;

