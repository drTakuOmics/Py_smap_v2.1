function outref=rot90j(inref,varargin);

nToRot=0;
if( nargin>1 )
    nToRot=mod(varargin{1},4);
end;

edgeSize=size(inref,1);
if( abs(nToRot)>0 )
outref=rot90(inref,nToRot);
if( mod(edgeSize,2)==0 )
    if( nToRot==1 )
        shifts=[1 0];
    elseif( nToRot==2 )
        shifts=[1 1];
    elseif( nToRot==3 )
        shifts=[0 1];
    elseif( nToRot==4 )
        shifts=[0 0];
    else
        shifts=[0 0];
    end;
else
    shifts=[0 0];
end;
outref=circshift(outref,[shifts(1),shifts(2)]);
else
    outref=inref;
end;

