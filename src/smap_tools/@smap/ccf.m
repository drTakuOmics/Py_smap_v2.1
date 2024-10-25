function outref=ccf(imref,templateref);

% sr=size(imref,1);
% sc=size(imref,2);
% Npix=sr.*sc;
% 
% a=smap.nm(imref);
% %b=smap.nm(templateref);
% b=templateref-mean(templateref(:));
% %b=b./std(b(:));
% 
% if( size(b,1)<size(a,1) || size(b,2)<size(a,2) )
%     b=smap.extendj(b,[size(a,1),size(a,2)],0);
% end;
% 
% b=smap.nm(b);
% %b=b./0.0036;%216;
% 
% aF=smap.ftj(a);
% bF=smap.ftj(b);
% 
% outref=smap.iftj(aF.*conj(bF));
% % outref=outref./(sqrt(Npix));




%%

nTemplates=size(templateref,3);
fullX=size(imref,1); fullY=size(imref,2);
fullXY=fullX*fullY;

imref=(imref-mean(imref(:)))./std(imref(:));
imref_F=fftn(ifftshift(imref));%./sqrt(fullXY);
%v=sum(abs(imrefVec).^2,1,'native'); %wait(gdev);
%v=v./(fullX.*fullX); %wait(gdev);
%denom=sqrt(v./(fullX*fullX));
%imref_F=imref_F./denom; %wait(gdev);

outref=zeros(size(imref,1),size(imref,2),nTemplates); 
%ccOut=ones(size(imref,1),size(imref,2)).*-1000;
peaks=[];

%pause

for i=1:nTemplates
    temp=templateref(:,:,i);
    temp=(temp-median(temp(:)))./std(temp(:));
    template=single(smap.cropOrPad(temp,[fullX,fullY],0));
        
    template_F=fftn(ifftshift(template));%./sqrt(fullXY);%-mean(template(:))));
    
%     templateVec=template_F(:); %wait(gdev); % normalize
%     v=sum(abs(templateVec).^2,1,'native'); %wait(gdev);
%     v=v./(fullX.*fullX); %wait(gdev);
%     denom=sqrt(v./(fullX*fullX));
%     template_F=template_F./denom; %wait(gdev);
    
    templateVec=template_F(:);
    v=sum(abs(templateVec).^2,1,'native');
    v=v./fullXY;
    denom=sqrt(v./fullXY);
    template_F=template_F./denom;
    
    cc_F=imref_F.*conj(template_F);
    tempCC=real(fftshift(ifftn(cc_F)))./sqrt(fullXY);
    outref(:,:,i)=tempCC;

    peaks(i)=max(tempCC(:));
    if( mod(i,100)==0 )
        fprintf('%d...',i);
    end;
    
end;

