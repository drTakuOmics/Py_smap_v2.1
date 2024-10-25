function CTF=ctf(df,edgeSize,varargin);
% replace the constants with an input structure for the microscope
% function CTF=ctf(edgeSize);

if( nargin<3 )

try
    params=df % actually expecting a params_image structure
    df=params.defocus;
catch

global params % input parameters
end

[Cs,Cc,V_acc,deltaE,a_i]=deal(params.Cs,params.Cc,params.V_acc, ...
    params.deltaE,params.a_i);
pixelSize=params.aPerPix;
if( isfield(params,'F_abs') )
    F_abs=params.F_abs;
else
    F_abs=0;
end;

cc=smap.def_consts();
h=cc.h; %h = 6.626068e-34;
e=cc.q_e; %e = 1.60217646e-19;
c_v=cc.c; %c_v = 2.99792458e8;
V=V_acc;

lambda = cc.h./((2.*cc.m_e.*cc.q_e.*V_acc).*(1+(cc.q_e.*V_acc./(2.*cc.m_e.*(cc.c.^2))))).^0.5;

edgeSize=max(edgeSize);
dummyIm=ones(edgeSize,edgeSize,'single');

[k_2d,cp]=smap.getKs(dummyIm,pixelSize);

R=single(smap.rrj(dummyIm)).*(floor(edgeSize./2)./0.5);
vec=-R(1,cp):R(end,cp);

[X,Y]=meshgrid(vec,vec);
Y=-Y;

gIm=acos(repmat(vec,size(R,1),1)./R);
gIm=atan(Y./X);

alpha_g=zeros(edgeSize,edgeSize);
tVal=atan(abs(Y)./abs(X));
tVal(isnan(tVal(:)))=0;
inds=find((X(:)>=0)&(Y(:)>=0));
if( ~isempty(inds) )
    alpha_g(inds)=tVal(inds);
end;
inds=find((X(:)<0)&(Y(:)>=0));
if( ~isempty(inds) )
    alpha_g(inds)=pi-tVal(inds);
end;
inds=find((X(:)<0)&(Y(:)<0));
if( ~isempty(inds) )
    alpha_g(inds)=pi+tVal(inds);
end;
inds=find((X(:)>=0)&(Y(:)<0)); 
if( ~isempty(inds) )
    alpha_g(inds)=2*pi-tVal(inds);
end;
inds=find((X(:)==0)&(Y(:)==0));
if( ~isempty(inds) )
    alpha_g(inds)=tVal(inds);
end;


nCTFs=size(df,1);
CTF=[];

% % % used only to verify that compilation is including the correct directories (052121):
% %
% df(:,1)=1;
% df(:,2)=10;
% %
% %

for i=1:nCTFs
    df1=df(i,1).*1e-9;
    df2=df(i,2).*1e-9;
    alpha_ast=df(i,3);
    
    ddf=df1-df2;
    df_ast=0.5.*(df1+df2+ddf.*cos(2.*(alpha_g-alpha_ast)));
    
    %else
    %    df_ast=dummyIm.*df1;
    %end;
    
    freq=k_2d.*1e10;
    chi=(pi.*lambda.*freq.^2).*(df_ast-((Cs.*((lambda.^2)*(freq.^2))./2)));
    
    w1=F_abs;%0;
    w2=1-w1;
    CTF_temp=complex(w1.*sin(chi)-w2.*cos(chi),-w1.*cos(chi)-w2.*sin(chi));
    
    % apply a coherence envelope:
    termOne=-(((pi.*lambda.*(freq.^2).*Cc.*deltaE)./(4.*V_acc.*sqrt(log(2)))).^2);
    termTwo=-((pi.*Cs.*(lambda.^2).*(freq.^3)-pi.*df_ast.*freq).^2).*(a_i.^2)/(log(2));
    CTF(:,:,i)=CTF_temp.*exp(termOne).*exp(termTwo);
    
end;


else
    fprintf('invoked CTF generation with old parameter types...\n');



% function CTF=ctf(df,edgeSize,pixelSize,varargin);
% 
%
% CTF=ctf(df,edgeSize,pixelSize,varargin);


% formerly: function CTF=smap_ctf(df,edgeSize,pixelSize,varargin);%,varargin);
if( nargin<5 )
    envFlag=1;
else
    envFlag=varargin{1};
end;

if( nargin>5 )
    F_abs=varargin{3};
else
    F_abs=0;
end;

if( length(df)<3 )
    df=[df df 0]
end;

pixelSize=varargin{1};
pixSize=pixelSize*1e-10;
N=edgeSize;
dummyIm=ones(N,N,'single');
%df=df*1e-9;

Cs=2.7e-3
% Cs=1e-6; % for 060817 dataset (Krios 1)
Cc=2.7e-3;
% Cc=3.5e-3;
deltaE=0.7;      % Energy spread of the source [eV]
% deltaE=1.1;
diam_obj=350e-6;%105e-6;%45e-6;%140e-6;%100e-6;   % Diameter of objective aperture [m]
%diam_obj=15e-6;
foc=3.5e-3;   % Focal distance [m]
F_abs=0.07

cc=smap.def_consts();
h=cc.h; %h = 6.626068e-34;
e=cc.q_e; %e = 1.60217646e-19;
a_i=cc.a_i; %a_i=0.05e-3; % Illumination aperture [rad]
% a_i=0;
c_v=cc.c; %c_v = 2.99792458e8;

V=cc.V; %300e3; %V=1e6;
% V=120e3
% V=100e3;

m_e = 9.10938215e-31; % kg
lambda = h/sqrt(e*V*m_e*(e/m_e*V/c_v^2 + 2 ))
m=m_e + e*V/(c_v^2)


[k_2d,cp]=smap.getKs(dummyIm,pixelSize);

R=single(smap.rrj(dummyIm)).*(floor(N./2)./0.5);
vec=-R(1,cp):R(end,cp);

[X,Y]=meshgrid(vec,vec);
Y=-Y;

gIm=acos(repmat(vec,size(R,1),1)./R);
gIm=atan(Y./X);

alpha_g=zeros(N,N);
tVal=atan(abs(Y)./abs(X));
tVal(isnan(tVal(:)))=0;
inds=find((X(:)>=0)&(Y(:)>=0));
if( ~isempty(inds) )
    alpha_g(inds)=tVal(inds);
end;
inds=find((X(:)<0)&(Y(:)>=0));
if( ~isempty(inds) )
    alpha_g(inds)=pi-tVal(inds);
end;
inds=find((X(:)<0)&(Y(:)<0));
if( ~isempty(inds) )
    alpha_g(inds)=pi+tVal(inds);
end;
inds=find((X(:)>=0)&(Y(:)<0)); 
if( ~isempty(inds) )
    alpha_g(inds)=2*pi-tVal(inds);
end;
inds=find((X(:)==0)&(Y(:)==0));
if( ~isempty(inds) )
    alpha_g(inds)=tVal(inds);
end;


nCTFs=size(df,1);
CTF=[];
for i=1:nCTFs
    df1=df(i,1).*1e-9;
    df2=df(i,2).*1e-9;
    alpha_ast=df(i,3);
    
    ddf=df1-df2;
    %if( alpha_ast~=0 )
    df_ast=0.5.*(df1+df2+ddf.*cos(2.*(alpha_g-alpha_ast)));
    %else
    %    df_ast=dummyIm.*df1;
    %end;
    
    freq=k_2d.*1e10;
    
    chi=(pi.*lambda.*freq.^2).*(df_ast-((Cs.*((lambda.^2)*(freq.^2))./2)));
    
    %CTF=exp(1i*chi); % old standard (up first)
    
    %w1=0.075;
    w1=F_abs;%0;
    w2=1-w1;
    CTF(:,:,i)=complex(w1.*sin(chi)-w2.*cos(chi),-w1.*cos(chi)-w2.*sin(chi));
    
    if( envFlag==1 )
        termOne=-(((pi.*lambda.*(freq.^2).*Cc.*deltaE)./(4.*V.*sqrt(log(2)))).^2);
        termTwo=-((pi.*Cs.*(lambda.^2).*(freq.^3)-pi.*df_ast.*freq).^2).*(a_i.^2)/(log(2));
        CTF(:,:,i)=CTF(:,:,i).*exp(termOne).*exp(termTwo);
    end;

end;

end;



