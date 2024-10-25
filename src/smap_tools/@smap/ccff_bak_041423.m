function [ccOut,peaks]=ccff(imref,templateref,varargin);
% ccOut=ccff(imref_F,template,'filt | noFilt');
ccMode='filt';
ntt=16;
if( nargin>2 )
    ccMode=varargin{1};
    if( nargin>3 )
        ntt=varargin{2};
    end;
end;

imref=(imref-mean(imref(:)))./std(imref(:));
switch ccMode
    case 'filt'
        if( size(imref,3)>1 & size(imref,3)==size(imref,2) )            
            fprintf('3D whitening...\n');
            nDims=3;
            dims_mat=[1 1 1];
            
            %pause;
            imref_F=smap.ftj(imref);
            fPSD=smap.psdFilter_3d(abs(imref_F));
            imFilt=smap.iftj(imref_F.*fPSD);
            nTemplates=size(templateref,4);
        else
            nDims=2;
            dims_mat=[1 1];
            [fPSD,imFilt,~]=smap.psdFilter(imref,'sqrt');
            imFilt=smap.nm(imFilt);
            nTemplates=size(templateref,3);
        end;
        
    case 'noFilt'
        imFilt=imref; fPSD=ones(size(imref));
    case 'ew'
        imFilt=extraWhiten(imref,ntt); fPSD=ones(size(imref));
end;

fullX=size(imref,1);

imref_F=fftn(ifftshift(imFilt))./sqrt(fullX.^nDims);
%imref_F=fftshift(smap.ftj(imFilt));
%imrefVec=imref_F(:); %wait(gdev); % normalize

%pause;

% % fix this:
% v=sum(abs(imrefVec).^2,1,'native'); %wait(gdev);
% v=v./(fullX.^nDims); %wait(gdev);
% denom=sqrt(v./(fullX.^nDims));
% imref_F=imref_F./denom; %wait(gdev);
%imref_F=imref_F./std(imref_F(:));

fPSD=ifftshift(fPSD);

if( nDims==2 )
    ccOut=zeros(size(imref,1),size(imref,2),nTemplates); 
elseif( nDims==3 )
    ccOut=zeros(cat(2,fullX.*dims_mat,nTemplates));
end;

peaks=[];

switch nDims
    case 2
        
        for i=1:nTemplates
            temp=templateref(:,:,i);
            
            switch ccMode
                case 'filt'
                    template=single(smap.cropOrPad(temp,fullX.*dims_mat,median(temp(:))));
                    template=smap.nm(template);
                    template=single(smap.cropOrPad(temp,fullX.*dims_mat,median(temp(:))));
            end;
            
            
            template_F=fftn(ifftshift(template))./(fullX.^nDims);%-mean(template(:))));
            
            switch ccMode
                case 'filt'
                    template_F=template_F.*fPSD;
                otherwise
            end;
            
%             templateVec=template_F(:); %wait(gdev); % normalize
%             v=sum(abs(templateVec).^2,1,'native'); %wait(gdev);
%             v=v./(fullX.^nDims); %wait(gdev);
%             denom=sqrt(v./(fullX.^nDims));
%             template_F=template_F./denom; %wait(gdev);
           template_F=template_F./std(template_F(:));

%            zzzz
            % convolve the image with the reversed template:
            cc_F=imref_F.*conj(template_F);
            tempCC=real(fftshift(ifftn(cc_F))).*(sqrt(fullX.^nDims));
            %    figure(3); clf; imsc(tempCC,[1560,2805],200); pause;
            ccOut(:,:,i)=gather(tempCC);
            %pause;
            
            %inds=tempCC>ccOut;
            %ccOut(inds)=tempCC(inds);
            peaks(i)=gather(max(tempCC(:)));
            if( mod(i,100)==0 )
                fprintf('%d...',i);
            end;
        end;

    case 3
        for i=1:nTemplates
            temp=templateref(:,:,:,i);
            
                    template=single(smap.cropOrPad(temp,fullX.*dims_mat,median(temp(:))));
                    template=smap.nm(template);
                    template=single(smap.cropOrPad(temp,fullX.*dims_mat,median(temp(:))));
            
            
            template_F=fftn(ifftshift(template));%-mean(template(:))));
            switch ccMode
                case 'filt'
                    template_F=template_F.*fPSD;
                otherwise
            end;
            
            %pause;
            
%             templateVec=template_F(:); %wait(gdev); % normalize
%             v=sum(abs(templateVec).^2,1,'native'); %wait(gdev);
%             v=v./(fullX.^nDims); %wait(gdev);
%             denom=sqrt(v./(fullX.^nDims));
%             template_F=template_F./denom; %wait(gdev);
            
            template_F=template_F./std(template_F(:));
            std(template_F(:));
            std(imref_F(:));
            
            % convolve the image with the reversed template:
            cc_F=imref_F.*conj(template_F);
            tempCC=real(fftshift(ifftn(cc_F))).*(sqrt(fullX.^nDims));
            
            %pause;
            
%                 temp=temp./std(temp(:));
%     cc_F=temp.*structref_F; % fast
%     %cc_F=complex(V_Fr_temp,V_Fi_temp).*vol_F; % fast
%     cc=fftshift(real(ifftn(ifftshift(cc_F)))).*sqrt(Npix.^3);
% 
            
            ccOut(:,:,:,i)=tempCC;
            
            %pause;
            
            %inds=tempCC>ccOut;
            %ccOut(inds)=tempCC(inds);
            peaks(i)=max(tempCC(:));
            if( mod(i,100)==0 )
                fprintf('%d...',i);
            end;
        end
        
end;

%fprintf('\n');

% %%
% function [outref]=extraWhiten(imref,varargin);
% % imref is unfiltered
% 
% patchSize=16;
% if( nargin>1 )
%     patchSize=varargin{1}
% end;
% im=imref;
% %[fPSD,im]=psdFilter(imref,'sqrt');
% 
% nPatches = (size(im,1)/patchSize)^2 %256;%400;  % (MAKE SURE SQUARE)
% imSize = patchSize^2;%256;
% patches = zeros(patchSize*patchSize,nPatches);
% patchIm = zeros(sqrt(nPatches)*patchSize);
% patchIm=im;
% 
% % PAD IMAGE FOR EDGE EFFECTS
% %im = padarray(im,[patchSize,patchSize],'symmetric');
%  
% inc=patchSize;
% ctr=1;
% % EXTRACT PATCHES...
% for iP = 1:sqrt(nPatches)
%     rows=[inc*(iP-1)+1:inc*(iP)];
%     for jP=1:sqrt(nPatches)
%         cols=[inc*(jP-1)+1:inc*(jP)];
%         tmp = im(rows,cols);
%         patches(:,ctr) = reshape(tmp,patchSize*patchSize,1);
% %         rowIdx = (ceil(ctr/sqrt(nPatches)) - 1)*patchSize + ...
% %             1:ceil(ctr/sqrt(nPatches))*patchSize;
% %         colIdx = (mod(jP-1,sqrt(nPatches)))*patchSize+1:patchSize* ...
% %             ((mod(jP-1,sqrt(nPatches)))+1);
% %         patchIm(rowIdx,colIdx) = tmp;
%         ctr=ctr+1;
%     end;    
% end
% 
% % CENTER IMAGE PATCHES
% patchesCentered = bsxfun(@minus,patches,mean(patches,2));
% 
% % %%
% % CALCULATE COVARIANCE MATRIX
% S = patchesCentered*patchesCentered'/nPatches; 
%  
% % DETERMINE EIGENECTORS & EIGENVALUES
% % OF COVARIANCE MATRIX
% [E,D] = eig(S);
%  
% % CALCULATE D^(-1/2)
% d = diag(D);
% d = real(d.^-.5);
% D = diag(d);
%  
% % CALCULATE WHITENING TRANSFORM
% W = E*D*E';
% 
% % %%
% % WHITEN THE PATCHES
% patchesWhitened = W*patchesCentered;
%  
% % DISPLAY THE WHITENED PATCHES
% wPatchIm = zeros(size(patchIm));
% for iP = 1:nPatches
%     rowIdx = (ceil(iP/sqrt(nPatches)) - 1)*patchSize + 1:ceil(iP/sqrt(nPatches))*patchSize;
%     colIdx = (mod(iP-1,sqrt(nPatches)))*patchSize+1:patchSize* ...
%              ((mod(iP-1,sqrt(nPatches)))+1);
%     wPatchIm(rowIdx,colIdx) = reshape(patchesWhitened(:,iP),...
%                                       [patchSize,patchSize]);
% end
%  
% outref=(wPatchIm-mean(wPatchIm(:)))./std(wPatchIm(:));;


