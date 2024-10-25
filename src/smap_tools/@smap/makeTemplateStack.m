function [ti,templateIm]=makeTemplateStack(nfIm,varargin);%templates,ss);

if( nargin>3 )
    refTemplateStack=varargin{3};
    disp('using coords from reference stack...');
else
    refTemplateStack=[];
    disp('making full-frame template stack from new ccs...');
end;

if( nargin>1 )
    
    templates=varargin{1};
    %ss=varargin{2};
    padVal=nanmedian(templates(:));
    ti=ones(size(nfIm,1),size(nfIm,2),'single').*padVal;%,length(rInds));
    templateIm=ones(size(nfIm,1),size(nfIm,2),size(templates,3),'single').*padVal;
    
    nPix=size(templates,1);
    for j=1:size(templates,3)
        temp=templates(:,:,j);
        if( isempty(refTemplateStack)==1 )
            % using phase shifts:
            cc=smap.ccf(nfIm,temp);
            [xt yt]=find(cc==max(cc(:)));
            nxny=smap.maxInterpF(cc,10,20,[xt yt]);
            tempPadded=smap.applyPhaseShifts(single(smap.extendj(temp,[2048,2048],'symmetric',padVal)),[-nxny(2) -nxny(1)]);
            templateIm(:,:,j)=single(smap.cutj(tempPadded,[size(nfIm,2),size(nfIm,1)]));
            ti=ti+templateIm(:,:,j);
        else            
            cc=smap.ccf(nfIm,refTemplateStack(:,:,j));
            [xt yt]=find(cc==max(cc(:)));
            nxny=smap.maxInterpF(cc,10,20,[xt yt]);
%             nxny=smap.maxInterpF(smap.ccf(nfIm,refTemplateStack(:,:,j)));
            tempPadded=smap.applyPhaseShifts(single(smap.extendj(temp,[2048,2048],'symmetric',padVal)),[-nxny(2) -nxny(1)]);
            templateIm(:,:,j)=single(smap.cutj(tempPadded,[size(nfIm,2),size(nfIm,1)]));
            ti=ti+templateIm(:,:,j);
        end;
        
        
        
    end;
        
    %ti=nm(sum(templateIm,3));
else
    disp('no templates...');
    templateIm=zeros(size(nfIm,1),size(nfIm,2));
    ti=zeros(size(nfIm,1),size(nfIm,2));
end;

