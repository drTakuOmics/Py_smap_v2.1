function gOut=g2(XYZ,varargin);
% gOut=g2(XYZ,beta);
%
% XYZ: 1-D, 2-D, or 3-D coordinate matrix
% beta=[amp sigma]
%

A=1; sigma=0.5; %x0=0;
if( nargin>1 )
    beta=varargin{1};
    A = beta(1) ; 
    sigma = beta(2) ; 
    %x0 = beta(3) ; 
end;

gOut=A.*exp((-(XYZ).^2)./(2.*(sigma.^2)));

