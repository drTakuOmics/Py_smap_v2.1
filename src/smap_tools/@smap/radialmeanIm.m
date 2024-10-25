function outref=radialmeanIm(imref,varargin);

sf=[]; opt=[];
%opt='circ';
if( nargin>2 )
    sf=varargin{2}; % smoothing factor to apply to sectors (if sectors are used)
    if( nargin>3 )
        opt=varargin{3};
    end;
end;

imref=gather(imref);
Npix=size(imref);
cp=floor(Npix./2)+1;
Npix_long=max(Npix); Npix_short=min(Npix);
cp_long=max(cp); cp_short=min(cp);
rCoord=smap.rrj(ones(Npix)).*Npix_long;

oddFlag=mod(size(imref,1),2);
if( oddFlag==1 )
    if( strcmp(opt,'circ') )
        rBins=[0:cp_long]./(Npix_long-1);
    else
        rBins=linspace(0,sqrt(2)./2,(Npix_long.*sqrt(2)./2)+1);
    end;
else
    if( strcmp(opt,'circ') )
        rBins=linspace(0,0.5,(Npix_long./2)+1);
    else
        rBins=linspace(0,sqrt(2)./2,((Npix_long+1).*sqrt(2)./2)+1);
        rBins=rBins(1:end-1);
    end;
end;

rBins=rBins.*Npix_long;

outref=zeros(1,length(rBins)-1,'single');
cpVal=imref(cp(1),cp(2));

outref=bindata((double(imref(:))),(rCoord(:)),(rBins))';

outref(1)=cpVal;
outref_Nd=[];

    
    
    
if( min(size(imref))==1 )
    edgeSize=floor((length(imref)+1)*2/sqrt(2))
    if( nargin>1 )
        edgeSize=varargin{1}
    end;
    
    realR=imref;
    RRd=single(smap.rrj(zeros(edgeSize,edgeSize)));
    inc=(max(RRd(:)))./(length(realR)-1)
    
    dummyR=[0:inc:max(RRd(:))];
    outref=zeros(size(RRd));
    for i=1:size(RRd,1)
        inds=find(isnan(RRd(i,:))==0);
        outref(i,inds)=interp1(dummyR,realR,RRd(i,inds),'linear');
    end;
    
    [ntfx2,ntfy2]=find(isnan(RRd)==1);
    if( ~isempty(ntfx2) )
        edgeVal=realR(round(size(imref,1)/2)-1);
        for i=1:length(ntfx2)
            outref(ntfx2(i),ntfy2(i))=edgeVal;%mean(realR(2:3));
        end;
    end;
    
else
    RRd=single(smap.rrj(imref));%(size(imref,1),size(imref,2),'freq'));
    if( size(imref,3)==1 )
        realR=single(smap.radialmeanj(imref));
        %Rt=RRd(:); Rt=Rt(find(Rt>0));
        inc=(max(RRd(:)))./(length(realR)-1);
        dummyR=[0:inc:max(RRd(:))];
        outref=zeros(size(RRd));
        for i=1:size(RRd,1)
            inds=find(isnan(RRd(i,:))==0);
            outref(i,inds)=interp1(dummyR,realR,RRd(i,inds),'linear');
        end;
        
        [ntfx2,ntfy2]=find(isnan(RRd)==1);
        if( ~isempty(ntfx2) )
            edgeVal=realR(round(size(imref,1)/2)-1);
            for i=1:length(ntfx2)
                outref(ntfx2(i),ntfy2(i))=edgeVal;%mean(realR(2:3));
            end;
        end;
        
    else
        imref=gather(imref);
        Npix=size(imref,1);
        cp=floor(Npix./2)+1;
        rBins=unique(abs(diag(RRd(:,:,cp))));
        rBins=sort([rBins; rBins+mean(diff(rBins))./2]);
        rmOut=zeros(1,length(rBins)-1,'single');
        rmOut(1)=gather(imref(cp,cp,cp));
        
        RRd=RRd(:);
        imref=imref(:);
        outref=nan(Npix,Npix,Npix,'single');
        
        for i=2:length(rBins)-1
            btu=[rBins(i) rBins(i+1)];
            bins=(RRd>=btu(1)) & (RRd<btu(2));
            rmOut(i)=smap.mean(imref(bins));
            outref(bins)=rmOut(i);
        end;
        outref(isnan(outref)==1)=rmOut(i);
        
    end;
end;


% %%
% function outref=radialmeanIm(imref);
% 
% if( min(size(imref))==1 )
%     %edgeSize=ceil(length(imref)*2/sqrt(2));
%     edgeSize=floor((length(imref)+1)*2/sqrt(2))
%     realR=imref;
%     RRd=single(smap.rrj(zeros(edgeSize,edgeSize)));
% else
%     RRd=single(smap.rrj(imref));%(size(imref,1),size(imref,2),'freq'));
%     realR=single(smap.radialmeanj(imref));
% end;
% 
% %Rt=RRd(:); Rt=Rt(find(Rt>0));
% inc=(max(RRd(:)))./(length(realR)-1);
% dummyR=[0:inc:max(RRd(:))];
% outref=zeros(size(RRd));
% for i=1:size(RRd,1)
%     inds=find(isnan(RRd(i,:))==0);
%     outref(i,inds)=interp1(dummyR,realR,RRd(i,inds),'linear');
% end;
% 
% [ntfx2,ntfy2]=find(isnan(RRd)==1);
% if( ~isempty(ntfx2) )
%     edgeVal=realR(round(size(imref,1)/2)-1);
%     for i=1:length(ntfx2)
%         outref(ntfx2(i),ntfy2(i))=edgeVal;%mean(realR(2:3));
%     end;
% end;