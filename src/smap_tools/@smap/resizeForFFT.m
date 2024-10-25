function outref=resizeForFFT(inref,varargin);

resize_mode='crop';
pad_val=0;
if( nargin>1 )
    try
        resize_mode=varargin{1};
    catch
        fprintf('problem reading crop/pad spec (using crop)...\n');
    end;
    if( nargin>2 )
        try
            pad_val=varargin{2};
        end;
    end;
end;

switch resize_mode
    case 'crop'
        inc=-1;
    case 'pad'
        inc=1;
    otherwise
        inc=-1;
end;

dims_orig=size(inref);
dims_new=[];
for i=1:length(dims_orig)
    dim=dims_orig(i);
% dim=max(size(inref));
    fprintf('old dim is %d pixels...',dim);
    while 1
        f=factor(dim);
        if( ~sum(f>3) )
            break;
        end;
        dim=dim+inc;
    end;
    dims_new(i)=dim;
end;

dims_new=max(dims_new).*ones(1,length(dims_orig));
meanVal=nanmean(inref(:));
inref=inref-meanVal;
% outref=cutj(inref,dims_new);
switch resize_mode
    case 'crop'
        outref=smap.cropOrPad(inref,dims_new);
    case 'pad'
        outref=smap.cropOrPad(inref,dims_new,pad_val);
    otherwise
end;
outref=outref+meanVal;

fprintf('new dim is %d pixels.\n',dim);

