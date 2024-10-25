function outref=radialmaxj(imref,varargin);
% % not yet implemented
% % use smap.rTheta instead
% % 
fprintf('*** not yet implemented *** \n');
outref=imref;

% %pause;
% 
% sf=[]; opt=[];
% if( nargin>2 )
%     sf=varargin{2}; % smoothing factor to apply to sectors (if sectors are used)
%     if( nargin>3 )
%         opt=varargin{3};
%     end;
% end;
% 
% % % 1D => 2D: this works, but needs to be adapted to use fast binning (081217)
% if( min(size(imref))==1 )
%     
%     edgeSize=length(imref).*2;
%     %if( mod(length(imref),2)==1 )
%     edgeSize=edgeSize-1;
%     %end;
%     realR=imref;
%     RRd=single(smap.rrj(zeros(edgeSize,edgeSize)));
%     %if( mod(length(imref),2)==1 )
%     RRd=RRd.*(edgeSize-1);
%     %else
%     %    RRd=RRd.*(edgeSize);
%     %end;
%     
%     cp=floor(edgeSize./2)+1;
%     inc=1;
%     dummyR=[0:inc:length(imref)-1];
%     outref=zeros(size(RRd));
%     for i=1:size(RRd,1)
%         inds=find(isnan(RRd(i,:))==0);
%         outref(i,inds)=interp1(dummyR,realR,RRd(i,inds),'linear');
%     end;
%     
%     [ntfx2,ntfy2]=find(isnan(RRd)==1);
%     if( ~isempty(ntfx2) )
%         edgeVal=realR(round(size(imref,1)/2)-1);
%         for i=1:length(ntfx2)
%             outref(ntfx2(i),ntfy2(i))=edgeVal;%mean(realR(2:3));
%         end;
%     end;
%     
%     lastVal=imref(end);
%     outref(find(isnan(outref(:))==1))=lastVal;
%     
% else
%     
%     Npix=size(imref,1);
%     rCoord=smap.rrj(imref).*Npix;
%     cp=smap.getcp(imref);
%     oddFlag=mod(size(imref,1),2);
%     if( oddFlag==1 )
%         if( strcmp(opt,'circ') )
%             %        rBins=linspace(0,0.5,(size(imref,1)./2)+1) %pause;
%             rBins=[0:((size(imref,1)./2)+1)]./(size(imref,1)-1);
%         else
%             rBins=linspace(0,sqrt(2)./2,(size(imref,1).*sqrt(2)./2)+1);
%         end;
%     else
%         if( strcmp(opt,'circ') )
%             rBins=linspace(0,0.5,(size(imref,1)./2)+1);
%         else
%             rBins=linspace(0,sqrt(2)./2,((size(imref,1)+1).*sqrt(2)./2)+1);
%             rBins=rBins(1:end-1);
%         end;
%         %rBins=rBins(1:end-1);
%     end;
%     
%     rBins=rBins.*Npix;
%     % rBins=rBins(1:2:end);
%     
%     outref=zeros(1,length(rBins)-1,'single');
%     cpVal=imref(cp(1),cp(2));
%     
%     outref=bindata(double(imref(:)),rCoord(:),rBins)';
%     
%     outref(1)=cpVal;
%     outref_2d=[];
%     
%     if( nargin>1 )
%         if( ~isempty(varargin{1}) )
%             n_tBins=varargin{1}; % # of theta bins
%             tBins=linspace(0,2.*pi,n_tBins); % check spacing near zero
%             t_inc=2*pi./n_tBins;
%             tBins=[0:t_inc:2*pi];
%             
%             
%             % % make second vector to bin:
%             R=single(smap.rrj(imref)).*(floor(Npix./2)./0.5);
%             
%             vec=-R(1,cp(1)):R(end,cp(1));
%             [X,Y]=meshgrid(vec,vec);
%             Y=-Y;
%             
%             % % % so instead we do this:
%             alpha_g=(double(atan((Y)./(X)))+pi/2)*2;
%             alpha_g(cp(1),cp(2))=-1;%nan;
%             
%             rtBins{2}=rBins;
%             rtBins{1}=tBins;
%             rCoord=complex(alpha_g,rCoord);
%             rCoord=rCoord(:);
%             
%             [test,y,y_new]=bindata(double(imref(:)),rCoord,rtBins,sf);
%             
%             outref_2d=reshape(y_new,Npix,Npix);
%             
%             outref_2d(cp(1),cp(1))=cpVal; % check this
%             
%         end;
%     end;
%     
% end;
% 
% function [ym,yb,y_full] = bindata(y,x,xrg,varargin);
% 
% sf=[];
% if( nargin>3 )
%     if( ~isempty(varargin{1}) )
%         sf=varargin{1};
%     end;
% end;
% 
% %function [ym,yb] = bindata(y,x,xrg)
% %Computes ym(ii) = mean(y(x>=xrg(ii) & x < xrg(ii+1)) for every ii
% %using a fast algorithm which uses no looping
% %If a bin is empty it returns nan for that bin
% %Also returns yb, the approximation of y using binning (useful for r^2
% %calculations). Example:
% %By Patrick Mineault
% %Refs: https://xcorr.net/?p=3326
% %      http://www-pord.ucsd.edu/~matlab/bin.htm
% 
% if( ~iscell(xrg) ) % 1-D output only
%     [~,whichedge] = histc(x,xrg(:)');
%     bins = min(max(whichedge,1),length(xrg)-1);
%     xpos = ones(size(bins,1),1);
%     ns = sparse(bins,xpos,1);
%     ysum = sparse(bins,xpos,y);
%     ym = full(ysum)./(full(ns));
%     yb = ym(bins);
%     y_full=[];
%     
% else
%     
%     x1=real(x);
%     x2=imag(x);
%     x1rg=xrg{1}(:);
%     x2rg=xrg{2}(:);
%     
%     [~,whichedge1] = histc(x1,x1rg(:)');
%     [~,whichedge2] = histc(x2,x2rg(:)');
%     
%     bins1 = min(max(whichedge1,1),length(x1rg)-1);
%     bins2 = min(max(whichedge2,1),length(x2rg)-1);
%     
%     bins = (bins2-1)*(length(x1rg)-1)+bins1;
%     
%     xpos = ones(size(bins,1),1);
%     ns = sparse(bins,xpos,1,(length(x1rg)-1)*(length(x2rg)-1),1);
%     ysum = sparse(bins,xpos,y,(length(x1rg)-1)*(length(x2rg)-1),1);
%     ym = full(ysum)./(full(ns));
%     
%     if( ~isempty(sf) )
%         ym=sgolayfilt(double(ym),1,sf);
%     end;
%     y_full=ym(bins);
%     yb = ym(bins);
%     ym = reshape(ym,length(x1rg)-1,length(x2rg)-1);
%     %    pause;
%     
% end;
% 
% 
% % % % original functions before sector averaging
% % function outref=radialmeanj(imref);
% %
% % Npix=size(imref,1);
% % rCoord=smap.rrj(imref).*Npix;
% % cp=smap.getcp(imref);
% % oddFlag=mod(size(imref,1),2);
% % if( oddFlag==1 )
% %     rBins=linspace(0,sqrt(2)./2,(size(imref,1).*sqrt(2)./2)+1);
% % else
% %     rBins=linspace(0,sqrt(2)./2,((size(imref,1)+1).*sqrt(2)./2)+1);
% %     rBins=rBins(1:end-1);
% % end;
% %
% % rBins=rBins.*Npix;
% % outref=zeros(1,length(rBins)-1,'single');
% % cpVal=imref(cp(1),cp(2));
% %
% % rCoord=rCoord(:);
% % imref=double(imref(:));
% % outref=bindata(imref,rCoord,rBins)';
% %
% % outref(1)=cpVal;
% %
% % function [ym,yb] = bindata(y,x,xrg)
% % %function [ym,yb] = bindata(y,x,xrg)
% % %Computes ym(ii) = mean(y(x>=xrg(ii) & x < xrg(ii+1)) for every ii
% % %using a fast algorithm which uses no looping
% % %If a bin is empty it returns nan for that bin
% % %Also returns yb, the approximation of y using binning (useful for r^2
% % %calculations). Example:
% % %By Patrick Mineault
% % %Refs: https://xcorr.net/?p=3326
% % %      http://www-pord.ucsd.edu/~matlab/bin.htm
% %
% % [~,whichedge] = histc(x,xrg(:)');
% % bins = min(max(whichedge,1),length(xrg)-1);
% % xpos = ones(size(bins,1),1);
% % ns = sparse(bins,xpos,1);
% % ysum = sparse(bins,xpos,y);
% % ym = full(ysum)./(full(ns));
% % yb = ym(bins);
% 



