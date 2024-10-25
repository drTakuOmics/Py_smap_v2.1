function [xs,ys,YS]=plotSH(peakVals,varargin);

ctu='b';
peakVals=peakVals(:);
xs=linspace(min(peakVals(:)),max(peakVals(:)),1e4);
% if( nargin>2 )
%     xs=varargin{2};
% end;


% xs=-7:1e-3:7;
Nsamples=length(peakVals(:));
if(nargin>1)
    ctu=varargin{1};
    if( nargin>2 )
        Nsamples=varargin{2};
        % if each peak was taken from a larger number:
%         samplesPerSample=varargin{2};
%         Nsamples=Nsamples.*samplesPerSample; 
% %         xs=varargin{2};        
    end;
end;

nfp=1;

N=hist(peakVals(:),xs);
ys=sum(N)-cumsum(N);

YS=(erfc(xs./sqrt(2))./2).*Nsamples;
thr=sqrt(2).*erfcinv(nfp.*2./(Nsamples))
nAbove=length(find(peakVals(:)>thr));

semilogy(xs,ys,ctu); hold on;
semilogy(xs,YS,'k--'); hold on;

%semilogy((xs.^2),sqrt(YS),'k--'); hold on;
%semilogy((xs.^2),sqrt(ys),ctu);

xl=[min(xs) max(xs).*1.01];
tempx=get(gca,'XLim');
if( min(tempx)<min(xs) )
    xl(1)=min(tempx);
end
if( max(tempx)>max(xs) )
    xl(2)=max(tempx);
end;

xlim(gather(xl)); %xlim([0 0.1]);
try
    ylim([9e-1 gather(2*max(YS))]);
catch
    fprintf('problem setting ylim...\n');
end;
% ahl(ndbavl(thr);
%title([num2str(thr) ' [' num2str(nAbove) ']']); 
axis square; grid on;
%pause(0.01);
