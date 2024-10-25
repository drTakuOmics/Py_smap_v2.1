function [ss,qBest,sI,xy]=clusterImByThr(imRef,mipiRef,thr,rotationsQ,varargin);

qFlag=0;
if( isa(rotationsQ,'quaternion') )
    qFlag=1;
end;

BW=imRef>thr;
mipVNT=imRef.*BW;
[nx,ny]=find(mipVNT>0); %disp(length(nx));
if( length(nx)>0 )    
    s=regionprops(BW,'area','PixelList');
    peakSum=[]; peakPix=[];
    for j=1:length(s)
        temp=[];
        for k=1:s(j).Area
            temp(k)=imRef(s(j).PixelList(k,2),s(j).PixelList(k,1));
        end;
        peakSum(j)=max(temp);
        peakPixInd=find(temp==max(temp),1,'first');
        peakPix(j,:)=[s(j).PixelList(peakPixInd,2) s(j).PixelList(peakPixInd,1)];
    end;
    
    ctr=1; 
    if( qFlag )
        qBest=quaternion.eye(1);
    else
        qBest=[];
    end;
    ss=struct('Area',{},'PixelList',{},'MaxVal',{},'q',{},'xy',{});
    mipMax=[]; mipiMax=[]; smv=[]; xy=[];
    for j=1:length(s)
        if( s(j).Area> 0 )%1 )
            mipMax(j)=0;
            mipiMax(j)=1;
            for k=1:size((s(j).PixelList),1)
                mipTemp=imRef(s(j).PixelList(k,2),s(j).PixelList(k,1));
                if( mipTemp>max(mipMax(j)) )
                    mipMax(j)=mipTemp;
                    mipiMax(j)=mipiRef(s(j).PixelList(k,2),s(j).PixelList(k,1));
                    ss(ctr).xy=[s(j).PixelList(k,2) s(j).PixelList(k,1)];
                    xy(ctr,:)=ss(ctr).xy;
                end;
            end;
            indToUse=mipiMax(j);
            if( qFlag )
                ss(ctr).q=rotationsQ(indToUse);
                qBest(ctr)=rotationsQ(indToUse);
            else
                ss(ctr).q=rotationsQ(:,:,indToUse);
                qBest(:,:,ctr)=rotationsQ(:,:,indToUse);
            end;
            ss(ctr).Area=s(j).Area;
            ss(ctr).PixelList=s(j).PixelList;
            ss(ctr).MaxVal=peakSum(j);
            smv(ctr)=ss(ctr).MaxVal;
            ctr=ctr+1;
        end
    end;
    [~,sI]=sort(smv,'descend');
    ss=ss(sI);
    xy=xy(sI,:);
    if( qFlag )
        qBest=qBest(sI);
    else
        qBest=qBest(:,:,sI);
    end;
else
    disp('no vals above thr');
    ss=[];
    qBest=[];
    sI=[];
    xy=[];
end;
