function outref=padForFFT(varargin);

inref=zeros(2,2);
method='image';
if( nargin>0 )
    inref=varargin{1};
    if( nargin>1 )
        method=varargin{2};
    end
end;

dim=max(size(inref));

fprintf('old dim is %d pixels...',dim);
while 1
    f=factor(dim);
    if( ~sum(f>3) )
%     if( sum(f<4 | f==5)==length(f) )
        break;
    end;
    dim=dim-1;
end;

switch method
    case 'image'
%         outref=smap.extendj(inref,[dim,dim],0);%smap.mean(inref(:)));
        outref=smap.cutj(inref,[dim,dim]);%smap.mean(inref(:)));
    case 'filter'
%         outref=smap.extendj(inref,[dim,dim],0);
        outref=smap.cutj(inref,[dim,dim]);
    otherwise
%         fprintf('unknown method...using median value\n');
%         outref=smap.extendj(inref,[dim,dim],nanmedian(inref(:)));
        outref=smap.cutj(inref,[dim,dim]);
end;

fprintf('new dim is %d pixels.\n',dim);

