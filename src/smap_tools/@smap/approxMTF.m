function MTFout=approxMTF(sfs,inputParams,varargin);
% 300 kV => ~0.5 MTF at f_nyq (Ruskin, Fig. 4); counting mode, 3 e/pix-s
% nb that MTF is only minimally changed by 5 e/pix-s (see between 3 eps and 7 eps in Fig. 5)
% also similar to Fig. 2 in McMullan 2014 (0.55 at 1 e/pix-s)
% this model works out correctly provided we use spatial frequencies that
% are treated as fraction of nyquist (although Rullgard claims that they
% are in units of nm^-1).
% 



% Pp=1;
% Pq=1;
% if( nargin>2 )
%     Pp=varargin{1};
%     Pq=varargin{2};
% end;

binFactor=2;
if( nargin>2 )
    binFactor=varargin{1};
end;

[k_arb,cp]=smap.getKs(size(sfs,1),0.5);
sfs=k_arb;

a=inputParams(1);
b=inputParams(2);
c=inputParams(3);
alpha=inputParams(4);
beta=inputParams(5);


MTFout=(a./(1+alpha.*sfs.^2))+(b./(1+beta.*sfs.^2))+c;

if( binFactor ~= 1 )
    MTFout=smap.resize_F(MTFout,binFactor,'fixedSize');%./binFactor;
end;
