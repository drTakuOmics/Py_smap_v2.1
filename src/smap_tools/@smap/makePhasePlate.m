function ppOut=makePhasePlate(imref,varargin);

Npix=size(imref,1);
cp=floor(size(imref,1)./2)+1;

method='vulovic';
k_cuton=Npix/100;
if( nargin>1 )
    method=varargin{1};
    if( nargin>2 )
        k_cuton=varargin{2};
    end;
end;

switch method
    case 'vulovic'
        
        k2d=smap.rrj(zeros(Npix,Npix)).*Npix;
%         k_cuton=floor(Npix/100);
        
        pp=ones(Npix,Npix).*exp(1i.*pi./2);
        pp(k2d<=k_cuton)=1;
        m=smap.cosMask(Npix,[(1./k_cuton).*0.5 (1./k_cuton).*0.8],0.25);
        
        pp_re=smap.iftj(smap.ftj(real(pp))).*m;
        pp_im=1-pp_re;

        pp=pp_re+1i*pp_im;

    case 'denk'
        
        k2d=smap.rrj(zeros(Npix,Npix)).*Npix;
%         k_cuton=floor(Npix/100);
        k_cuton=Npix/50;
        pp=ones(Npix,Npix).*exp(1i.*pi./2);
        pp(k2d<=k_cuton)=1;        
        m=smap.cosMask(Npix,[(1./k_cuton).*0.5 (1./k_cuton).*0.8],0.25);

%         pp_im=smap.iftj(smap.ftj(1-imag(pp)));
%         pp_re=smap.iftj(smap.ftj(1-real(pp)));
        pp_im=smap.iftj(smap.ftj(1-imag(pp))).*m;
        pp_re=1-pp_im;
%         pp_re=smap.iftj(smap.ftj(1-real(pp)));
        pp=pp_re+1i*pp_im;

end;


ppOut=pp;
