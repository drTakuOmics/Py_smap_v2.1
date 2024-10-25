function outref=rrj(inref);
% get the k-space coordinates 
% inref is any matrix that is the size of the k-space

dims_orig=size(inref);
dim_max=max(size(inref));
dims=[size(inref,1) size(inref,2) size(inref,3)];
Npix=dim_max;%size(inref,1);
cp=floor(Npix./2)+1;
x=[1:Npix]-cp;

X=zeros(dim_max,dim_max,dims(3));
for i=1:dims(3)
    X(:,:,i)=repmat(x,dim_max,1);
end;

if( dims(3)>1 )
    Y=permute(X,[2 1 3]);
    Z=zeros(size(Y));
    cp_z=floor(dims(3)./2)+1;
    for i=1:size(Z,3)
        Z(:,:,i)=cp_z-i;
    end;
%     Z=permute(X,[3 1 2]);
    R=sqrt(X.^2+Y.^2+Z.^2);
    outref=R./(2.*R(1,cp,cp));%Npix;
else
    Y=permute(X,[2 1]);
    R=sqrt(X.^2+Y.^2);
    outref=R./(2.*R(cp,1));
end;

if( sum(abs(size(X)-dims_orig))>0 )
    outref=smap.cropOrPad(outref,dims_orig);
end;

% dims=[size(inref,1) size(inref,2) size(inref,3)];
% Npix=size(inref,1);
% cp=smap.getcp(inref);
% cp=cp(1);
% x=[1:Npix]-cp;
% 
% X=zeros(dims(1),dims(2),dims(3));
% for i=1:dims(3)
%     X(:,:,i)=repmat(x,dims(1),1);
% end;
% 
% if( dims(3)>1 )
%     Y=permute(X,[2 1 3]);
%     Z=permute(X,[3 1 2]);
%     R=sqrt(X.^2+Y.^2+Z.^2);
%     outref=R./(2.*R(1,cp,cp));%Npix;
% else
%     Y=permute(X,[2 1]);
%     R=sqrt(X.^2+Y.^2);
%     outref=R./(2.*R(cp,1));
% end;
