function occref_out=occ(imref,templateref,varargin);

if( ~isreal(imref) )
    imref_F=imref;
else
    imref_F=smap.ftj(imref);
end;

if( ~isreal(templateref) )
    templateref_F=templateref;
else
    templateref=templateref-mode(templateref(:));
    if( size(templateref,1)<size(imref,1) )
        templateref=smap.extendj(templateref,size(imref,1).*[1,1,1],0);
    end;
    templateref_F=smap.ftj(templateref);
end;

if( nargin>2 )
    q2=varargin{1};
else
    q2=ones(size(imref));
end;

N=size(imref,1);
cp=floor(N./2)+1;
nDims=length(find(size(imref)>0));
pix_factor=sqrt(N.^nDims);

denom_F=abs(templateref_F(:)).^2;
denom_F_sum=sum(q2(:).*denom_F(:))./pix_factor;

cc_F=imref_F.*conj(templateref_F);
cc=real(ifftn(fftshift(cc_F)));
cc=ifftshift(cc);%idx_f{:});
occref_out=cc.*(pix_factor./denom_F_sum);

