%% 
function filtref=psdFilter_3d(inref,varargin);

Npix_here=size(inref,1);
cp_here=floor(Npix_here./2)+1;

rCoord=smap.rrj(inref).*Npix_here;

inds=find(rCoord<=cp_here);
%rBins=linspace(0,0.5,2.*cp_here).*Npix_here;

oddFlag=mod(size(inref,1),2);
if( oddFlag==1 )
    rBins=[0:((size(inref,1)./2)+1)]./(size(inref,1)-1);
else
    rBins=linspace(0,0.5,(size(inref,1)./2)+1);
end;
rBins=rBins.*Npix_here;


[test,y,y_new]=smap.bindata((double(inref(inds))),(rCoord(inds)),(rBins));

%pause;

temp=nan(Npix_here,Npix_here,Npix_here);
temp(inds)=y;
inds_shell=find((cp_here-rCoord)<2 & (cp_here-rCoord)>=0 );
inds=find(rCoord>cp_here);
temp(inds)=nanmean(temp(inds_shell));
temp(cp_here,cp_here,cp_here)=1;

filtref=1./temp;
filtref(cp_here,cp_here,cp_here)=0;

