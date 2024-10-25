function outref = mask_central_cross(imref,varargin);

Npix=size(imref,1);
cp=floor(Npix./2)+1;
temp = ones(Npix,Npix,class(imref));

ift_flag=0;
if( isreal(imref) )
    imref_F=smap.ftj(imref);
    ift_flag=1;
else
    imref_F=imref;
end;

dc_val = abs(imref_F(cp,cp));

imref_F(cp,:)=0;
imref_F(:,cp)=0;


% m2=mean(abs(imref_F),2);
% s2=2*std(abs(imref_F),[],2);
% inds_sub=find(abs(imref_F(:,cp))>(m2+2.*s2));
% imref_F(inds_sub,cp)=m2(inds_sub);
% m2=mean(abs(imref_F),2);
% s2=2*std(abs(imref_F),[],2);
% inds_sub=find(abs(imref_F(:,cp-1))>(m2+2.*s2));
% imref_F(inds_sub,cp-1)=m2(inds_sub);
% m2=mean(abs(imref_F),2);
% s2=2*std(abs(imref_F),[],2);
% inds_sub=find(abs(imref_F(:,cp+1))>(m2+2.*s2));
% imref_F(inds_sub,cp+1)=m2(inds_sub);

% m2=mean(abs(imref_F),1);
% s2=2*std(abs(imref_F),[],1);
% inds_sub=find(abs(imref_F(cp,:))>(m2+2.*s2));
% imref_F(cp,inds_sub)=m2(inds_sub);
% m2=mean(abs(imref_F),1);
% s2=2*std(abs(imref_F),[],1);
% inds_sub=find(abs(imref_F(cp-1,:))>(m2+2.*s2));
% imref_F(cp-1,inds_sub)=m2(inds_sub);
% m2=mean(abs(imref_F),1);
% s2=2*std(abs(imref_F),[],1);
% inds_sub=find(abs(imref_F(cp+1,:))>(m2+2.*s2));
% imref_F(cp+1,inds_sub)=m2(inds_sub);

imref_F(cp,cp)=dc_val;

if( ift_flag )
    outref=smap.iftj(imref_F);
else
    outref=imref_F;
end;

