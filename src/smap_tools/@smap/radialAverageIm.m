function outref=radialAverageIm(imref);

RRd=single(smap.rrj(imref));%(size(imref,1),size(imref,2),'freq'));
realR=single(smap.radialmeanj(imref));
Rt=RRd(:); Rt=Rt(find(Rt>0));
inc=(max(RRd(:)))./(length(realR)-1);
dummyR=[0:inc:max(RRd(:))]; 
outref=zeros(size(RRd));
for i=1:size(RRd,1)
    inds=find(isnan(RRd(i,:))==0);
    outref(i,inds)=interp1(dummyR,realR,RRd(i,inds),'linear');
end;

[ntfx2,ntfy2]=find(isnan(RRd)==1);
edgeVal=realR(round(size(imref,1)/2)-1);
for i=1:length(ntfx2)
    outref(ntfx2(i),ntfy2(i))=edgeVal;%mean(realR(2:3));
end;

