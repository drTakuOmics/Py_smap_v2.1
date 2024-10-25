function outref=ipcc(imref,templateref);
% outref=ipcc(imref,templateref);
% 

N=size(imref,1);
cp=floor(N./2)+1;

% if( isreal(imref) )
%     imref_N=imref-mean(imref(:));
%     imref_N=imref_N./std(imref_N(:));
%     imref_F=smap.ftj(imref_N); 
% %     imref_F=log(abs(imref_F));
% %     imref_F=log(abs(imref_F-min(abs(imref_F(:)))));
%     imref_F=abs(imref_F);
% end;
% if( isreal(templateref) )
%     templateref_N=templateref-median(templateref(:));
%     templateref_N=templateref_N./std(templateref_N(:));
%     templateref_N=smap.extendj(templateref_N,[N,N],0);
%     templateref_N=templateref_N-mean(templateref_N(:));
%     templateref_N=templateref_N./std(templateref_N(:));
%     templateref_F=smap.ftj(templateref_N); 
%     templateref_F=abs(templateref_F);
% end;
% 
% % % % put a PSD filter in here:
% % [pFilt,imOut]=smap.psdFilter(imref_F,'sqrt');
% % imref_F_filt=imOut;
% % templateref_F_filt=smap.applyFilter(templateref_F,pFilt);
% imref_F_filt=imref_F;
% templateref_F_filt=templateref_F;

imref_F_filt=imref;
templateref_F_filt=templateref;

% Transform the high passed FFT phase to Log Polar space
imref_p = smap.polarImage(imref_F_filt, N, N, N,N, 'bicubic', [cp cp], 'valid');
templateref_p = smap.polarImage(templateref_F_filt, N, N, N,N, 'bicubic', [cp cp] , 'valid');
imref_p=imref_p-mean(imref_p(:));
imref_p=imref_p./std(imref_p(:));
templateref_p=templateref_p-mean(templateref_p(:));
templateref_p=templateref_p./std(templateref_p(:));

% Convert log polar magnitude spectrum to FFT
imref_pF = smap.ftj(imref_p);
templateref_pF = smap.ftj(templateref_p);

% cc=((imref_pF).*conj(templateref_pF))./abs((imref_pF).*conj(templateref_pF));
cc=((imref_pF).*conj(templateref_pF));
outref=smap.iftj(cc);
for j=1:size(outref,1)
    outref(j,:)=outref(j,:)-mean(outref(j,:));
    outref(j,:)=outref(j,:)./std(outref(j,:));
end;

% [I,J]=find(outref==max(outref(:)),1,'first');
% 
% theta_s = sort(outref(:));  % TODO speed-up, we surely don't need to sort
% SI = length(theta_s):-1:(length(theta_s));
% [theta_x,theta_y] = find(outref == theta_s(SI));
% % Compute angle of rotation
% DPP = 360 / size(outref, 2);
% Theta = DPP * (theta_y - 1);




% %%
% 
% tt_F=smap.ftj(smap.nm(tt));
% tt_F_r=real(tt_F);
% tt_F_i=imag(tt_F);
% %tt_p_r=smap.rTheta(tt_F_r);
% %tt_p_i=smap.rTheta(tt_F_i);
% tt_p_r = smap.polarImage(tt_F_r,N,N,N,N,'bicubic',[cp cp],'valid');
% tt_p_i = smap.polarImage(tt_F_r,N,N,N,N,'bicubic',[cp cp],'valid');
% 
% %%
% 
% %RR=RotationMatrix(quaternion.eulerangles('xyz',[0 0 25.*pi./180]))
% 
% %nfIm=smap.nm(poissrndj(tt.*20));
% nfIm=smap.nm(poissrndj(circshift(tt,[20,20]).*20));
% %nfIm=smap.nm(poissrndj(rot90(tt,0).*50));
% nfIm=smap.nm(randn(N,N));
% 
% N=size(nfIm,1);
% cp=floor(size(N)./2)+1;
% 
% nfIm_F=smap.ftj(nfIm);
% nfIm_F_r=real(nfIm_F);
% nfIm_F_i=imag(nfIm_F);
% 
% %im_p_r=smap.rTheta(nfIm_F_r);
% %im_p_i=smap.rTheta(nfIm_F_i);
% im_p_r = smap.polarImage(nfIm_F_r,N,N,N,N,'bicubic',[cp cp],'valid');
% im_p_i = smap.polarImage(nfIm_F_i,N,N,N,N,'bicubic',[cp cp],'valid');
% 
% 
% im_p_F=im_p_r+1i.*im_p_i;
% tt_p_F=tt_p_r+1i.*tt_p_i;
% 
% cc_p_F=im_p_F.*conj(tt_p_F);
% 
% cc=smap.iftj(cc_p_F);
% imsc(cc);
% 









%%
if (1==0)
    
    load ~/matching/10.03.14/17/2W0O/230nm/10.03.14_FOV17_2W0O_230nm_mip_out.mat ss
    
    for i=1:length(ss)
        q(i)=ss(i).q;
    end;
    
    resetSeed
    qf=quaternion.randRot(1,17);
 
    
    tt=smap_templates(q,[210.6191  240.9534   -0.3390],400,0.965,'2W0O');
    ttf=smap_templates(qf,[210.6191  240.9534   -0.3390],400,0.965,'2W0O');
    
    %%
    
    clear z;
    for i=1:length(q)
        z(:,:,i)=smap.ipcc(a,tt(:,:,i));
        zf(:,:,i)=smap.ipcc(a,ttf(:,:,i));
        disp(i);
        figure(203);
%         imsc(z(:,:,i)); title(i); drawnow;
        plot(max(z(:,:,i),[],1),'b'); title(i); drawnow; hold on;
        plot(max(zf(:,:,i),[],1),'r'); title(i); drawnow; hold on;
    end;
    
    
    
end;














