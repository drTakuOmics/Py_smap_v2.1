function outref=rTheta(imref);

RRd=single(smap.rrj(imref));
N=size(imref,1);

% [k_2d,cp]=smap.getKs(imref,1);
cp=floor((N./2)+1);
% N=size(imref,1);
R=single(smap.rrj(imref)).*(floor(N./2)./0.5);
vec=-R(1,cp):R(end,cp);

[X,Y]=meshgrid(vec,vec);
Y=-Y;

% gIm=acos(repmat(vec,size(R,1),1)./R);
% gIm=atan(Y./X);

alpha_g=zeros(N,N);
for i=1:N
    for j=1:N
        tVal=atan(abs(Y(i,j))./abs(X(i,j)));
        if( (X(i,j)>=0)&&(Y(i,j)>=0) )
            alpha_g(i,j)=tVal;
        elseif( (X(i,j)<0)&&(Y(i,j)>=0) )
            alpha_g(i,j)=pi-tVal;
        elseif( (X(i,j)<0)&&(Y(i,j)<0) )
            alpha_g(i,j)=pi+tVal;
        elseif( (X(i,j)>=0)&&(Y(i,j)<0) )
            alpha_g(i,j)=2*pi-tVal;
        end;
        if( (X(i,j)==0)&&(Y(i,j)==0) )
            alpha_g(i,j)=0;
        end;
    end;
end;

% alpha_g(RRd>RRd(cp,end))=nan;
% RRd(RRd>RRd(cp,end))=nan;

Npix=size(imref,1);
rCoord=smap.rrj(imref).*Npix;

tCoord=alpha_g;
cp=getcp(imref);
oddFlag=mod(size(imref,1),2);
if( oddFlag==1 )
    rBins=linspace(0,sqrt(2)./2,(size(imref,1).*sqrt(2)./2)+1);
%     rBins=logspace(0,1,(size(imref,1).*sqrt(2)./2)+1);
else
    rBins=linspace(0,sqrt(2)./2,((size(imref,1)+1).*sqrt(2)./2)+1);
%     rBins=logspace(0,1,((size(imref,1)+1).*sqrt(2)./2)+1);
    rBins=rBins(1:end-1);
end;

%tBins=0:pi./80:2.*pi;%0:5:360;
tBins=[0:359].*pi./180;
rBins=rBins.*Npix;

% rCoord=smap.rrj(imref);
% rCoord=rCoord./max(rCoord(:)); rCoord=log((rCoord.*9)+1);
% rBins=smap.radialmeanj(rCoord);

outref=zeros(length(tBins)-1,length(rBins)-1,'single');
rCoord=rCoord(:);
tCoord=tCoord(:);
imref=imref(:);

for i=2:length(rBins)-1
    btu_r=[rBins(i) rBins(i+1)];
    for j=2:length(tBins)-1
        btu_t=[tBins(j) tBins(j+1)];
        bins=(rCoord>=btu_r(1)) & (rCoord<btu_r(2) ) & (tCoord>=btu_t(1)) & (tCoord<btu_t(2) );
        outref(j,i)=smap.mean(imref(bins));
    end;
    if( mod(i,10)==0 )
        disp(i);
    end;
end;

meanVal=smap.mean(outref(:));
outref(isnan(outref))=meanVal;
outref=smap.nm(outref);





