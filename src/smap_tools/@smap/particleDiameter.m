function outref=particleDiameter(volref,varargin);
% estimate particle diameter
% varargin is the fraction-of-max threshold (between 0 and 1)
%
% function outref=particleDiameter(volref,varargin)

thresh=0.005;%1/exp(2);
if( nargin>1 )
    thresh=varargin{1};
end;

Npix=max(size(volref));
rCoord=smap.rrj(volref).*Npix;

cp=floor(Npix./2)+1;
oddFlag=mod(size(volref,1),2);
if( oddFlag==1 )
    rBins=linspace(0,sqrt(2)./2,(size(volref,1).*sqrt(2)./2)+1);
else
    rBins=linspace(0,sqrt(2)./2,((size(volref,1)+1).*sqrt(2)./2)+1);
    rBins=rBins(1:end-1);
end;

rBins=rBins.*Npix;
%outref=zeros(1,length(rBins)-1,'single');
cpVal=volref(cp,cp,cp);

rCoord=rCoord(:);
volref=double(volref(:));
binnedVals=bindata(volref,rCoord,rBins)';
%binnedVals(1)=cpVal;
%binnedVals=binnedVals-min(binnedVals);
binnedVals=binnedVals-median(binnedVals);
binnedVals=binnedVals./max(binnedVals);
outref=find(binnedVals>thresh,1,'last').*2;

%pause;

% function outref=particleDiameter(volref,varargin);
% % estimate particle diameter
% % varargin is the fraction-of-max threshold (between 0 and 1)
% % function outref=particleDiameter(volref,varargin)
% thresh=0.005;%1/exp(2);
% if( nargin>1 )
%     thresh=varargin{1};
% end;
% 
% Npix=max(size(volref));
% rCoord=smap.rrj(volref).*Npix;
% 
% cp=floor(Npix./2)+1;
% oddFlag=mod(size(volref,1),2);
% if( oddFlag==1 )
%     rBins=linspace(0,sqrt(2)./2,(size(volref,1).*sqrt(2)./2)+1);
% else
%     rBins=linspace(0,sqrt(2)./2,((size(volref,1)+1).*sqrt(2)./2)+1);
%     rBins=rBins(1:end-1);
% end;
% 
% rBins=rBins.*Npix;
% %outref=zeros(1,length(rBins)-1,'single');
% cpVal=volref(cp,cp,cp);
% 
% rCoord=rCoord(:);
% volref=double(volref(:));
% binnedVals=bindata(volref,rCoord,rBins)';
% binnedVals(1)=cpVal;
% binnedVals=binnedVals-min(binnedVals);
% binnedVals=binnedVals./max(binnedVals);
% outref=find(binnedVals>thresh,1,'last').*2;
% %outref=find(binnedVals>min(binnedVals),1,'last').*2;

% thresh=1/exp(2);
% if( nargin>1 )
%     thresh=varargin{1};
% end;
% 
% volref=abs(volref);
% 
% if( size(volref,3)>1 )
% proj=smap.projView(volref);
% else
%     for j=1:3
%         proj(:,:,j)=smap.nm(volref);
%     end;
% end;
% 
% dim=max(size(proj));
% rj=zeros(dim,dim,3);
% for j=1:3
%     rj(:,:,j)=smap.radialmeanIm(proj(:,:,j));
% end;
% 
% mv=max(rj,[],3);
% 
% mp=max(mv,[],1);
% mp=mp-min(mp);
% mp=mp./max(mp);
% cp=smap.getcp(proj(:,:,1));
% x=[1:dim]-cp(1);
% 
% outref=sum(abs([x(find(mp(1:cp(1))>(1./exp(2)),1,'first')) x(cp(1)+find(mp(cp(1):end)<(1./exp(2)),1,'first'))]));


