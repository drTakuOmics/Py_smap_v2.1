function outref=getcp(inref);
% just gets the center pixel
nDims=length(size(inref));
outref=floor(size(inref)./2)+1;
%             outref=[floor(size(inref,1)./2)+1 floor(size(inref,2)./2)+1];
