function outref=rotate2dMatrix(inref,R,varargin);

%method='cpu';
%method='gpu';

%pause;

Npix=size(inref,1);
cp=gather(floor(Npix./2)+1);

if( isa(inref,'gpuArray') )
    method='gpu';
    V_Fr_out=gpuArray.zeros(Npix,Npix,'single');
    V_Fi_out=gpuArray.zeros(Npix,Npix,'single');
else
    method='cpu';
    V_Fr_out=zeros(Npix,Npix,'single');
    V_Fi_out=zeros(Npix,Npix,'single');
end;

if( isreal(inref) )
    inputType='realspace';
    inref=smap.ftj(inref);
    
else
    inputType='kspace';
end;



dcVal=inref(cp,cp);
inref(cp,cp)=0;
V_Fr=real(inref); V_Fi=imag(inref);
clear inref;
vec=([-((cp-1)):((cp-1))]);
if( mod(Npix,2)==0 )
    vec=vec(1:end-1);
end;

[xd,yd]=meshgrid(vec,vec);
xd=xd(:,1:Npix); yd=yd(:,1:Npix);
xy=([xd(:) yd(:)]);
clear xd yd zd;
if( size(R,1)==3 )
    R=R(1:2,1:2,:);
end;
xy_r=(xy*R); % must be double-precision
RR=(sqrt(sum(xy_r.^2,2)));
inds=find(RR<(cp-1));
xy_r=xy_r(inds,:)+cp;

clear RR;
%tic
X=xy_r(:,1);
Y=xy_r(:,2);
clear xy_r;

dummy=[1:Npix];
temp=interpn(dummy,dummy,V_Fr,Y,X); % 19 s
V_Fr_out(inds)=temp;
temp=interpn(dummy,dummy,V_Fi,Y,X); % 19 s
V_Fi_out(inds)=temp;
outref=complex(V_Fr_out,V_Fi_out);
outref(cp,cp)=dcVal;
%toc

if( strcmp(inputType,'realspace') )
    %fprintf('ifft...\n');
    outref=smap.iftj(outref);
    
end;
