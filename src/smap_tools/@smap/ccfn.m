function outref=ccfn(imref,templateref);

nTemplates=size(templateref,3);
fullX=size(imref,1); fullY=size(imref,2);
fullXY=fullX*fullY;
cp=floor(fullX./2)+1;

imref=(imref-mean(imref(:)));
%[fPSD,imFilt,~]=smap.psdFilter(imref,'sqrt');

fAmp=abs(fftshift(fftn(ifftshift(imref))))./fullX;
fAmp_r=smap.radialmeanIm(fAmp);
fAmp_r(cp,cp)=1;
fAmp_r_inv=1./fAmp_r;
fAmp_r_inv(cp,cp)=nan;
fAmp_r_inv(cp,cp)=nanmean(nanmean(fAmp_r_inv(cp-1:cp+1,cp-1:cp+1)));
fPSD=(fAmp_r_inv./sum(abs(fAmp_r_inv(:)).^2)).*(fullX.^2);

fPSD=ifftshift(fPSD);
imref_F=fftn(ifftshift(imref)).*(fPSD);
imrefVec=imref_F(:);
v=sum(abs(imrefVec).^2,1,'native');
v=v./fullXY;
denom=sqrt(v./fullXY); % divide by SD
%denom=(v./fullXY); % divide by variance; causes ccs to go down with increasing mean
imref_F=imref_F./denom;

outref=zeros(size(imref,1),size(imref,2),nTemplates); 
peaks=[];
for i=1:nTemplates
    temp=templateref(:,:,i);
    temp=(temp-mean(temp(:)));
    template=single(smap.extendj(temp,[fullX,fullX],0));
    template_F=fftn(ifftshift(template));
    template_F=template_F.*fPSD;
    
    cc_F=imref_F.*conj(template_F);
    tempCC=real(fftshift(ifftn(cc_F)));
    outref(:,:,i)=tempCC;
    
    peaks(i)=max(tempCC(:));
    if( mod(i,100)==0 )
        fprintf('%d...',i);
    end;
end;
