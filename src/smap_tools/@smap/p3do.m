function p3do(xyzref,varargin);
if( nargin<2 )
    ctu='b';
else
    ctu=varargin{1};
end;
plot3(xyzref(1,:),xyzref(2,:),xyzref(3,:),'o','LineWidth',2,'Color',ctu);



