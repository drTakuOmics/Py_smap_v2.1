function p3d(xyzref,varargin);

if( size(xyzref,1)~=3 )
    xyzref=xyzref';
end;

if( nargin>1 )
    ctu=varargin{1};
else
    ctu='k.';
end;

plot3(xyzref(1,:),xyzref(2,:),xyzref(3,:),ctu); axis equal; grid on; 
smap.p3a;
xlabel('x'); ylabel('y'); pause(0.01);

