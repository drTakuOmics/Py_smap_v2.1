function outref=ccfv(imref,templateref);
%


%%

nTemplates=size(templateref,3);
fullX=size(imref,1); fullY=size(imref,2);
fullXY=fullX*fullY;

imref=(imref-mean(imref(:)));%./var(imref(:));
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
    temp=(temp-mean(temp(:)));%./var(temp(:));
    template=single(smap.extendj(temp,[fullX,fullX],0));
    %template=template./var(template(:));
    template_F=fftshift(fftn(ifftshift(template)));%./sqrt(fullXY);%-mean(template(:))));
    
    templateVec=template_F(:);
    v=sum(abs(templateVec).^2,1,'native');
    v=v./fullXY;
    %denom=sqrt(v./fullXY);
    denom=(v./fullXY);    
    template_F=template_F./denom;
    
    cc_F=imref_F.*conj(template_F);
    tempCC=real(fftshift(ifftn(cc_F)))./(fullXY);
    outref(:,:,i)=tempCC;

    peaks(i)=max(tempCC(:));
    if( mod(i,100)==0 )
        fprintf('%d...',i);
    end;
    
end;

if( 0 )

%%
i=1;
temp=templateref(:,:,i);
temp=(temp-mean(temp(:)));%./var(temp(:));
template=single(smap.extendj(temp,[fullX,fullX],0));
template=template./var(template(:));
template_F=fftn(ifftshift(template));%./sqrt(fullXY);%-mean(template(:))));

templateVec=template_F(:);
v=sum(abs(templateVec).^2,1,'native');
v=v./fullXY;
%denom=sqrt(v./fullXY);
denom=(v./fullXY);
template_F=template_F./denom;

%%
% template 1:
a1=tt(:,:,1); 

% template 2:
a2=tt(:,:,2);

% subtract means, then compare dot products:
a1=a1-mean(a1(:));
a2=a2-mean(a2(:));
abs(dot(a1(:),a1(:))-dot(a2(:),a2(:)))./dot(a1(:),a1(:))
% ans: 0.4681

% additionally divide by SD, then compare dot products:
a1s=a1./std(a1(:));
a2s=a2./std(a2(:));
abs(dot(a1s(:),a1s(:))-dot(a2s(:),a2s(:)))./dot(a1s(:),a1s(:))
% ans: 0.88

% additionally divide by variance, then compare dot products:
a1v=a1./var(a1(:));
a2v=a2./var(a2(:));
abs(dot(a1v(:),a1v(:))-dot(a2v(:),a2v(:)))./dot(a1v(:),a1v(:))
% ans: 1.4e-13


%%
% take templates of a tubulin patch (4x3 dimers) at two
% orientations that change the footprint substantially - case 1 is looking
% down the MT axis (3 tubulins are visible in projection), and case 2 is
% orthogonal (12 tubulins are visible).

fprintf('\n\n\n   ***   \n\n\n');

% template 1 & 2:
template1=tt(:,:,1);
template2=tt(:,:,2);

% images (zero-mean):
a1=poissrndj(tt(:,:,1).*10);
a2=poissrndj(tt(:,:,2).*10);
a1=a1-mean(a1(:));
a2=a2-mean(a2(:));

% subtract template means, then compare dot products:
template1=template1-mean(template1(:));
template2=template2-mean(template2(:));
cc1=smap.iftj(smap.ftj(a1).*conj(smap.ftj(template1)));
cc1=cc1./std(cc1(:));
cc2=smap.iftj(smap.ftj(a2).*conj(smap.ftj(template2)));
cc2=cc2./std(cc2(:));
[abs(dot(a1(:),template1(:))-dot(a2(:),template2(:)))./dot(a1(:),template1(:)) ...
    (max(cc1(:))-max(cc2(:)))./max(cc1(:))]

% additionally divide by SD, then compare dot products:
template1s=template1./std(template1(:));
template2s=template2./std(template2(:));
cc1s=smap.iftj(smap.ftj(a1s).*conj(smap.ftj(template1s)));
cc1s=cc1s./std(cc1s(:));
cc2s=smap.iftj(smap.ftj(a2s).*conj(smap.ftj(template2s)));
cc2s=cc2s./std(cc2s(:));
[abs(dot(a1(:),template1s(:))-dot(a2(:),template2s(:)))./dot(a1(:),template1s(:)) ...
    (max(cc1s(:))-max(cc2s(:)))./max(cc1s(:))]

% additionally divide by variance, then compare dot products:
template1v=template1./var(template1(:));
template2v=template2./var(template2(:));
cc1v=smap.iftj(smap.ftj(a1).*conj(smap.ftj(template1v)));
cc1v=cc1v./std(cc1v(:));
cc2v=smap.iftj(smap.ftj(a2).*conj(smap.ftj(template2v)));
cc2v=cc2v./std(cc2v(:));
[abs(dot(a1(:),template1v(:))-dot(a2(:),template2v(:)))./dot(a1(:),template1v(:)) ...
    (max(cc1v(:))-max(cc2v(:)))./max(cc1v(:))]

% ans: fractional difference in [dot products, peak/SD]
% [0.5015    0.3174]
% [0.3165    0.3174]
% [0.0629    0.3174]
%
% i.e., normalizing by variance comes close to preserving the dot product,
% but the ratio of peak/SD in a single xcorr is identical.



end;
