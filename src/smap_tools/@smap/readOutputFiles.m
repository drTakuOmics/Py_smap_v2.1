function outref=readOutputFiles(obj,fileType);
% varargout=readOutputFiles(obj,fileType);
% % obj can be (obj or ID); fileType is a search.results property.
cObj=class(obj);
switch cObj
    case 'search'
        fn=smap.checkBaseDir(eval(['obj.results.' char(fileType)]))
        fileType=char(fileType);
    case 'char'
        ID=obj;
%         if( isempty(name) && isempty(ext) )
            baseDir=smap.checkBaseDir;
            dirToUse=[baseDir 'searches/'];
            load([dirToUse ID '.mat'],'s');
            obj=s;
            fn=smap.checkBaseDir(eval(['obj.results.' char(fileType)]));
%             fileType=char(fileTypeOrPath);
%         else
%             fn=char(smap.checkBaseDir(fileTypeOrPath));
%             fileType=char(fileTypeOrPath);
%         end;
    otherwise
        disp('not programmed');
end;


switch fileType
        
    case 'histogram'
        nBins=4096;
        nBins_dot=360;
        fid=fopen(fn,'r');
        classHist=fread(fid,inf,'int64')';
        fclose(fid);
        h=reshape(classHist,nBins,nBins_dot);
        binRange=obj.params.binRange;
        X=linspace(-binRange,binRange,nBins);
        hs=sum(h,2);
        sh_full=sum(hs)-cumsum(hs);
        
        outref.X=X; 
        outref.sh_full=sh_full; 
        outref.hs=hs; 
        outref.classHist=classHist;
        
    case 'above_thr_list'
        fid=fopen(fn,'r');
        temp=fread(fid,inf,'double');
        fclose(fid);
        arbVals=temp(1:3:end); 
        arbTemp=temp(2:3:end); 
        arbInds=temp(3:3:end);
        
        tempStruct=smap.readOutputFiles(obj,'mip_image');
        tempIm=tempStruct.mip_image;
        dummyVec=1:(size(tempIm,1)*size(tempIm,2));
        dummyIm=reshape(dummyVec,size(tempIm));
        temp=arbTemp; temp(find(temp==0))=1;
        [a,b]=ind2sub(size(dummyIm),dummyIm(temp));
        arbLocs=[a b];
        
        outref.arbVals=arbVals; 
        outref.arbLocs=arbLocs; 
        outref.arbInds=arbInds;
        
    case 'peaks_by_rotation_list'
        fid=fopen(fn,'r');
        temp=fread(fid,inf,'double');
        fclose(fid);
        peakVals=temp(1:2:end); 
        peakTemp=temp(2:2:end);
        
        tempStruct=smap.readOutputFiles(obj,'mip_image');
        tempIm=tempStruct.mip_image;
        dummyVec=1:(size(tempIm,1)*size(tempIm,2));
        dummyIm=reshape(dummyVec,size(tempIm));
        temp=peakTemp; temp(find(temp==0))=1;
        [a,b]=ind2sub(size(dummyIm),dummyIm(temp));
        peakLocs=[a b];
        
        outref.peakVals=peakVals;
        outref.peakLocs=peakLocs;

    case 'dp_list'
        fid=fopen(fn,'r');
        dots=fread(fid,inf,'double');
        fclose(fid);
        outref.dots=dots;

    case {'mip_image','sum_image','squared_sum_image','mipi_image','filt_image'}
        
        temp=smap.ri(fn);
        eval(['outref.' char(fileType) '=temp;']);
        
    otherwise
        disp('not programmed')
        outref=[];
        
end


% F=ezfit(X,sum(h,2),'gauss');
% figure(101); clf; plot(X,sum(h,2)); showfit(F);
% sig=F.m(2);
% Xn=X./sig;
% arbVals=arbVals./sig;
% peakVals=peakVals./sig;
% mip=mip./sig;
% 
% sh_full_s=sum(hs_s)-cumsum(hs_s);
% 
% Nsamples=sum(hs);
% ys=sum(hs)-cumsum(hs);
% YS=(erfc(Xn./sqrt(2))./2).*Nsamples;
% thr=sqrt(2).*erfcinv(1.*2./(Nsamples));
% 
% figure(104); clf;
% plot(Xn,sh_full,'b'); hold on;
% set(gca,'yscale','log');
% semilogy(Xn,YS,'k--'); hold on;
% xlim([-0.1 Xn(find(sh_full>0,1,'last'))]); ylim([0.8 2.*max(sh_full)]); grid on;
% 
% thrErf=sqrt(2).*erfcinv(1.*2./(size(nfIm,1).*size(nfIm,2).*length(rotationsQ)));
% avl(thrErf);
% xlabel('SNR'); ylabel('survival count'); axis square;
% pause(0.05);
% 
% XX=floor(classHist_hw.*((dots-params.dot_mean)./(2.*params.dot_std))+classHist_center);


%     ssImage=tr([pwd '/doneSquaredSumImage']);
%     meanImage=sImage./nSamples;
%     squaredMeanImage=ssImage./nSamples;
%     varImage=squaredMeanImage-(meanImage.^2);
%     mipNorm=(mipOrig-meanImage)./sqrt(varImage);





