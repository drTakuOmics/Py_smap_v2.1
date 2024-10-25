function [outref,shifts,peak]=reg2vols(inputVol,refVol,varargin);
% register 3D matrix #1 against 3D matrix #2
% assumes the refVol is larger
% [regVol,shifts]=reg2vols(inputVol,refVol,varargin);
%

if( nargin<3 )
    rtu=100;
else
    rtu=varargin{1};
end;

nr=round(size(refVol,1)/2);
nc=round(size(refVol,2)/2);
np=round(size(refVol,3)/2);

sr=size(refVol,1);
sc=size(refVol,2);
sp=size(refVol,3);

padFactor=max([sr sc sp]);

a=(inputVol-mean(inputVol(:)))./std(inputVol(:));
b=(refVol-mean(refVol(:)))./std(refVol(:)); clear refVol;

a=single(smap.extendj(a,size(permute(b,[2 1 3])),'symmetric',min(a(:))));

ppdataFT=fftshift(fftn(ifftshift(a)));
pptemplateFT=fftshift(fftn(ifftshift(b)));

outcc=single(fftshift(ifftn(ifftshift(ppdataFT.*conj(pptemplateFT)))));
outcc=(outcc-mean(outcc(:)))./std(outcc(:));

tempC=single(smap.cutj(outcc,[rtu,rtu,rtu]));

%[nx,ny,nz]=ind2sub(size(outcc),find(outcc==max(outcc(:))))
%[nx,ny,nz]=ind2sub(size(tempC),find(tempC==max(tempC(:))))
%nxnynz=[nr-nx nc-ny np-nz]

%regVol=circshift(inputVol,[nxnynz(1)+1,nxnynz(2)+1,nxnynz(3)+1]);
%regVol=single(smap.extendj(regVol,size(permute(b,[2 1 3])),'symmetric',min(inputVol(:))));

%tempC=single(smap.cutj(outcc,[rtu,rtu,rtu]));

dx=-1:0.001:1;
[xt,yt,zt]=ind2sub(size(tempC),find(tempC==max(tempC(:))));
peak=max(tempC(:));

lL=-1; uL=1;

uLx=1;
if( xt==1 )
    lLx=0;
    disp('warning - edge');
else
    lLx=-1;
end;
try
P=polyfit([lLx 0 uLx]',squeeze(tempC(xt-1:xt+1,yt,zt)),2);
A=P(1); B=P(2); C=P(3);
yy=A.*(dx.^2)+B.*(dx)+C;
itu=[(find(yy==max(yy),1,'first')) (find(yy==max(yy),1,'last'))];
itu=round((itu(1)+itu(2))./2);
nx=-((rtu./2)-(xt+dx(itu))+1);
catch
    nx=-((rtu./2)-xt);    
    disp('problem interpolating x');
end;

uLy=1;
if( yt==1 )
    lLy=0;
    disp('warning - edge');
else
    lLy=-1;
end;
try
P=polyfit([lLy 0 uLy],squeeze(tempC(xt,yt-1:yt+1,zt)),2);
A=P(1); B=P(2); C=P(3);
yy=A.*(dx.^2)+B.*(dx)+C;
itu=[(find(yy==max(yy),1,'first')) (find(yy==max(yy),1,'last'))];
itu=round((itu(1)+itu(2))./2);
ny=-((rtu./2)-(yt+dx(itu))+1);
catch
    ny=-((rtu./2)-yt);
    disp('problem interpolating y');
end;

uLz=1;
if( zt==1 )
    lLz=0;
    disp('warning - edge');
else
    lLz=-1;    
end;
try
P=polyfit([lLz 0 uLz],(squeeze(tempC(xt,yt,zt-1:zt+1)))',2);
A=P(1); B=P(2); C=P(3);
yy=A.*(dx.^2)+B.*(dx)+C;
itu=[(find(yy==max(yy),1,'first')) (find(yy==max(yy),1,'last'))];
itu=round((itu(1)+itu(2))./2);
nz=-((rtu./2)-(zt+dx(itu))+1);
catch
    nz=-((rtu./2)-zt);
    disp('problem interpolating z');
end;

shifts=[nx ny nz]
%nxny=[nx ny];
outref=smap.applyPhaseShifts(inputVol,[-nx -ny -nz]);



