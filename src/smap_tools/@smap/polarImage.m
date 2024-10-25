function [r,g,b] = polarImage(A, Ar, Ac, Nrho, Ntheta, Method, Center, Shape)
% [r,g,b] = polarImage(A, Ar, Ac, Nrho, Ntheta, Method, Center, Shape)
% from https://www.mathworks.com/matlabcentral/fileexchange/
% 19731-fourier-mellin-image-registration
%

% Inputs:   A       the input image
%           Nrho    the desired number of rows of transformed image
%           Ntheta  the desired number of columns of transformed image
%           Method  interpolation method (nearest,bilinear,bicubic)
%           Center  origin of input image
%           Shape   output size (full,valid)
%           Class   storage class of A

global rho;

theta = linspace(0,2*pi,Ntheta+1); theta(end) = [];

switch Shape
    case 'full'
        corners = [1 1;Ar 1;Ar Ac;1 Ac];
        d = max(sqrt(sum((repmat(Center(:)',4,1)-corners).^2,2)));
    case 'valid'
        %     d = min([Ac-Center(1) Center(1)-1 Ar-Center(2) Center(2)-1])
%         d=Center(1);
            d=Center(1);
end
minScale = 1;
% minScale=0.1;

% % switch between log and linear representations here (091116/jpr):
% default 'base 10' logspace - play with d to change the scale of the log axis
rho = logspace(log10(minScale),log10(d),Nrho)';
% rho=linspace(0,d,Nrho)';

% convert polar coordinates to cartesian coordinates and center
xx = rho*cos(theta) + Center(1);
yy = rho*sin(theta) + Center(2);
if strcmp(Method,'nearest'), % Nearest neighbor interpolation
    r=interp2(A,xx,yy,'nearest');
elseif strcmp(Method,'bilinear'), % Linear interpolation
    r=interp2(A,xx,yy,'linear');
elseif strcmp(Method,'bicubic'), % Cubic interpolation
    r=interp2(A,xx,yy,'cubic');
else
    error(['Unknown interpolation method: ',method]);
end

% any pixels outside warp, pad with black
mask= (xx>Ac) | (xx<1) | (yy>Ar) | (yy<1);
r(mask)=0;




% ep=600;
% r=r(ep:end,:);
% cp=smap.getcp(r);
% r=r(:,cp:end);
% % ep1=1; % up: removes low ks
% % ep2=200; % up: removes high ks
% % ep2=size(r,1)-ep2;
% % r=r(ep1:ep2,:);
% % figure(304);
% % imsc(r);































































