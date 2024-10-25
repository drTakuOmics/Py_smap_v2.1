function mask=makeFilt(imSize,maskEdges,aPerPix,varargin);

temp=max(size(imSize));
if( temp > 1 )
    nDims=length(find(size(imSize)>1))
    imSize=repmat(temp,1,nDims);
end;

maskEdgeIn=maskEdges(1); 
maskEdgeOut=maskEdges(2);

% nn=single(smap.rrj(ones(imSize,imSize,imSize)));
nn=single(smap.rrj(ones(imSize)));
R=nn./aPerPix;



Rt=(R<=maskEdgeIn);
RR=abs(R-maskEdgeIn).*(1-Rt);
Rtt=(R>maskEdgeOut);
RR(find(Rtt==1))=pi/2;
T=(maskEdgeIn-maskEdgeOut).*2;

RRR=0.5+0.5.*cos(2.*pi.*RR./T);
RRR(find(Rtt==1))=0;
mask=RRR;

