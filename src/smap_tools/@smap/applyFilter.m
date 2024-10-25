function imOut=applyFilter(imref,tm,varargin)
% assumes filter has origin to the right (and down) of true center for
% even-sized images; filter should indeed be origin-shifted

if( size(imref,3)==1 )
    if( nargin<3)
        normFlag=1;
    else
        normFlag=varargin{1};
    end;
    
    inDims=[size(imref,1) size(imref,2)];
    if( normFlag==1 )
        imref=(imref-mean(imref(:)))./std(imref(:));
    end;
    
    if( (size(imref,1)~=size(tm,1)) || (size(imref,2)~=size(tm,2)) )
        imref=single(smap.extendj(imref,[size(tm,1),size(tm,2)],mean(imref(:))));%'symmetric',mean(imref(:))));
        imref=single(smap.cutj(imref,[size(tm,1),size(tm,2)]));
    end;
    
%    cp=smap.getcp(tm);
%    tm(cp(1),cp(2))=1;

    % imrefF=fftshift(fft2(ifftshift(imref)));
    imrefF=smap.ftj(imref);
    
    % imOut=real(fftshift(ifft2(ifftshift(imrefF.*(tm)))));
    imOut=smap.iftj(imrefF.*tm);
    
    imOut=single(smap.cutj(imOut,[inDims(1),inDims(2)]));
    
    
else
    
    disp('will apply a 3D filter...');
    filtref=tm;
    Npix=size(imref,1);
    R=smap.rrj(imref).*Npix;
    
    % rCoord=smap_rrj(filtref).*Npix;
    rCoord=smap.rrj(filtref).*Npix;
    % cp=getcp(filtref);
    cp=smap.getcp(filtref);
    
    rBins=unique(abs(diag(rCoord)));
    rBins=sort([rBins; rBins+mean(diff(rBins))./2]);
    
    rmOut=zeros(1,length(rBins)-1,'single');
    rmOut(1)=filtref(cp(1),cp(2));
    
    rCoord=rCoord(:);
    filtref=filtref(:);
    filt3d=zeros(Npix,Npix,Npix,'single');
    
    for i=2:length(rBins)-1
        btu=[rBins(i) rBins(i+1)];
        bins=(rCoord>=btu(1)) & (rCoord<btu(2));
        %     rmOut(i)=smap_mean(filtref(bins));
        rmOut(i)=smap.mean(filtref(bins));
        bins3d=(R>=btu(1)) & (R<btu(2));
        filt3d(bins3d)=rmOut(i);
    end;
    
    filt3d(cp(1),cp(2),cp(1))=tm(cp(1),cp(2));
    % imOut=iftj(ftj(imref).*filt3d);
    imOut=smap.iftj(smap.ftj(imref).*filt3d);
    
end;
