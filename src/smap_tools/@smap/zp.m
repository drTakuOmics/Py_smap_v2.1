function zpOut=zp(numIn,N,varargin);

padChar='0';
if( nargin>2 )
    padChar=varargin{1};
end;

if( ischar(numIn)==0 )
    numIn=num2str(numIn);
end;

while( length(numIn)<N )
    numIn=[padChar numIn];
end;
zpOut=numIn;
