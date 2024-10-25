function outref=mean(inref,varargin);
dim=1;
if( nargin>1 )
    dim=varargin{1};
end;
x=inref(:);

nans = isnan(x);
x(nans) = 0;
n = sum(~nans,1);
outref = sum(x,1)./n;
