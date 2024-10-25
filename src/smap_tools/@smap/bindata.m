function [ym,yb,y_full] = bindata(y,x,xrg,varargin);

sf=[];
if( nargin>3 )
    if( ~isempty(varargin{1}) )
        sf=varargin{1};
    end;
end;

%function [ym,yb] = bindata(y,x,xrg)
%Computes ym(ii) = mean(y(x>=xrg(ii) & x < xrg(ii+1)) for every ii
%using a fast algorithm which uses no looping
%If a bin is empty it returns nan for that bin
%Also returns yb, the approximation of y using binning (useful for r^2
%calculations). Example:
%By Patrick Mineault
%Refs: https://xcorr.net/?p=3326
%      http://www-pord.ucsd.edu/~matlab/bin.htm

if( ~iscell(xrg) ) % 1-D output only
    [~,whichedge] = histc(x,xrg(:)');
    bins = min(max(whichedge,1),length(xrg)-1);
    xpos = ones(size(bins,1),1);
    ns = sparse(bins,xpos,1);
    ysum = sparse(bins,xpos,y);
    ym = full(ysum)./(full(ns));    
    yb = ym(bins);
%     yb = [ym(bins) bins];


    ym=ym(find(ns));
    y_full=xrg(find(ns));
    
else
    
    x1=real(x);
    x2=imag(x);
    x1rg=xrg{1}(:);
    x2rg=xrg{2}(:);
    
    [~,whichedge1] = histc(x1,x1rg(:)');
    [~,whichedge2] = histc(x2,x2rg(:)');
    
    bins1 = min(max(whichedge1,1),length(x1rg)-1);
    bins2 = min(max(whichedge2,1),length(x2rg)-1);
    
    bins = (bins2-1)*(length(x1rg)-1)+bins1;
    
    xpos = ones(size(bins,1),1);
    ns = sparse(bins,xpos,1,(length(x1rg)-1)*(length(x2rg)-1),1);
    ysum = sparse(bins,xpos,y,(length(x1rg)-1)*(length(x2rg)-1),1);
    ym = full(ysum)./(full(ns));
    
    if( ~isempty(sf) )
        ym=sgolayfilt(double(ym),1,sf);
    end;
    y_full=ym(bins);
    yb = ym(bins);
    ym = reshape(ym,length(x1rg)-1,length(x2rg)-1);
%    pause;
    
end

end


