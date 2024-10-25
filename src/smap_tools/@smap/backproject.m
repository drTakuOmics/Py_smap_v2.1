function [outref,dummy_noW,otherref]=backproject(patchref,Rref,varargin);


% slow_flag=0;
% slow_flag=1
global params

% params=[];
% params.Cs=0.000001;
% params.Cc=0.0027;
% params.V_acc=300e3;
% params.deltaE=0.7;
% params.a_i=50e-6;
% params.aPerPix=1.032;
% params.F_abs=0.07;

%aPerPix=1.032;
padSize=size(patchref,1);
inrefSize=size(patchref,1);
CTF=[]; otherref=[]; fPSD=[];
if( nargin>2 )
    %padSize=varargin{1};
    df=varargin{1};
%     CTF=imag(smap.ctf(df,padSize,aPerPix,0,0.0));
    CTF=imag(smap.ctf(df,padSize.*[1 1]));
else 
    fprintf('no CTF correction...\n');
end;

nPatches=size(patchref,3);
Npix=size(patchref,1);
cp=floor(Npix./2)+1;

fprintf('Calculating FFTs...\n');
p_F=zeros(padSize,padSize,nPatches,'single');
p_F=p_F+1i*p_F;
for i=1:nPatches
    temp=patchref(:,:,i);
    temp=smap.extendj(temp,padSize.*[1,1],mean(temp(:)));
    
    p_F(:,:,i)=smap.ftj(temp);
    if( ~isempty(CTF) )
        p_F(:,:,i)=p_F(:,:,i).*(CTF(:,:,i)); % CTF multiplication
        %p_F(:,:,i)=p_F(:,:,i).*sign(CTF(:,:,i)); % phase flipping
%         temp=smap.radialmeanIm(abs(p_F(:,:,i)));
%         temp(cp,cp)=1;
%         fPSD(:,:,i)=temp; 
%         fPSD(cp,cp,i)=0;
    end;
end;

T=1;
[x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
dummyX=1:Npix;
x0=x(:,:,(cp-T):(cp+T))-cp;
y0=y(:,:,(cp-T):(cp+T))-cp;
z0=zeros(Npix,Npix,2.*T+1);
for j=1:(2.*T+1)
    z0(:,:,j)=j;
end;
z0=z0-(2.*T+2)./2;
box=zeros(size(z0));
cp_box=floor(size(box,3)./2)+1;
        
% % % ewald correction:
% k_2d=smap.getKs(ones(Npix,Npix),1.032);
% k_ny=k_2d(cp,end);
% x0_temp=x0.*k_ny./cp;%max(abs(x0(cp,:)));
% y0_temp=y0.*k_ny./cp;%max(abs(y0(:,cp)));
% temp=smap.def_consts();
% V=temp.V;%300e3;
% %V=1e6;
% m_e = 9.10938215e-31; % kg
% lambda = temp.h/sqrt(temp.q_e*V*m_e*(temp.q_e/m_e*V/temp.c^2 + 2 ));
% lambda=lambda.*1e10;
% Rmax=(1./lambda).*aPerPix;
% z0=real(Rmax-sqrt((Rmax.^2)-(x0_temp.^2+y0_temp.^2)));
% z0=z0.*cp./k_ny;
% max(z0(:))

xyz=gpuArray(double([x0(:) y0(:) z0(:)]));
clear x0 y0 z0;

%pause;

% %%
W=(zeros(Npix,Npix,Npix,'single'));
WW=W;
dummy_F=(zeros(Npix,Npix,Npix,'single'));
dummy_F=dummy_F+1i*dummy_F;
otherref=W;

f_total=0; pix_total=0;


%pause;

% if( slow_flag )
%     dummy_temp_F=dummy_F;
%     nx=gpuArray(0); ny=nx; nz=nx;
% end;

% % this is fairly lame (nearest-neighbor interpolation)
% % now also uses what should be a windowed sinc weighting (grigorieff 2007)
fprintf('Adding slices...\n');
tic;
for i=1:nPatches
    
    %%
%     dummy_F=(zeros(Npix,Npix,Npix,'single'));
%     dummy_F=dummy_F+1i*dummy_F;
%     %RM=normalizeRM(RotationMatrix(quaternion.eulerangles('xyz',[30 60 0].*pi./180)));

    RM=Rref(:,:,i)';
    xyz_r=(RM*xyz')';
    frame=p_F(:,:,i);

% linear:

%pause;
    
    box(:,:,cp_box)=frame;
    temp=xyz_r-round(xyz_r);
    %xyz_r=round(xyz_r)+cp;
    xyz_r=xyz_r+cp;
    inds=find(xyz_r(:,1) > 1 & xyz_r(:,2)>1 & xyz_r(:,3)>1 & ...
        xyz_r(:,1)<Npix & xyz_r(:,2)<Npix & xyz_r(:,3)<Npix);
    xyz_r=xyz_r(inds,:);
    temp=temp(inds,:);
    xyz_r_round=round(xyz_r);
    %dist=xyz_r-xyz_r_round;
    dist=xyz_r_round-xyz_r;
    inds_w=sub2ind(padSize.*[1 1 1],xyz_r_round(:,2),xyz_r_round(:,1),xyz_r_round(:,3));
    
    X=xyz(inds,1)+cp+temp(:,1); 
    Y=xyz(inds,2)+cp+temp(:,2);
    Z=xyz_r(:,3)+cp;
    
%    newVals=complex(interpn(dummyX,dummyX,real(frame),Y,X),interpn(dummyX,dummyX,imag(frame),Y,X));
%    newVals(isnan(newVals(:)))=0;
%    dummy_F(inds_w)=dummy_F(inds_w)+gather(newVals);
    X=xyz(inds,1)+cp+dist(:,1); 
    Y=xyz(inds,2)+cp+dist(:,2); 
    Z=xyz(inds,3)+dist(:,3);
    if( T>0 )
        newVals=complex(interpn(1:Npix,1:Npix,-T:T,real(box),Y,X,Z),interpn(1:Npix,1:Npix,-T:T,imag(box),Y,X,Z));
    else
        newVals=complex(interpn(1:Npix,1:Npix,real(box),Y,X),interpn(1:Npix,1:Npix,imag(box),Y,X));
    end;
    newVals(isnan(newVals(:)))=0;
    dummy_F(inds_w)=dummy_F(inds_w)+gather(newVals);
    
%     if( slow_flag )
%         dummy_temp_F=dummy_temp_F.*0;
%         dummy_temp_F(inds_w)=gather(newVals);
%         temp_r=smap.iftj(gpuArray(dummy_temp_F));
%         ind_lin=find(temp_r==max(temp_r(:)),1,'first');
%         [nx(i),ny(i),nz(i)]=ind2sub(Npix.*[1 1 1],ind_lin);
%         plot3(nx,ny,nz,'o'); drawnow;
%     end;
    
    
% 
% % % diagnostic (test interpolation quality by rotating back):
%     dummy_r=zeros(Npix,Npix);
%     dummy_i=zeros(Npix,Npix);
%     itu=sub2ind([Npix Npix],xyz(inds,2)+cp,xyz(inds,1)+cp);
%     dummy_r(itu)=real(dummy_F(inds_w));
%     dummy_i(itu)=imag(dummy_F(inds_w));
%     dummy=complex(dummy_r,dummy_i);
%     
%     fsc=smap.radialmeanj(real((dummy.*conj(frame))./sqrt((abs(dummy).^2).*(abs(frame).^2))),[],[],'circ');
%     figure(303); clf;
%     plot(fsc);
%     grid on; ylim([0 1.05]);
%     dummy_lin=dummy;
%     
%         
%     dummy_F=(zeros(Npix,Npix,Npix,'single'));
%     dummy_F=dummy_F+1i*dummy_F;
%     
%     xyz_r=(RM*xyz')';
%     
%     %temp=sqrt(sum((xyz_r(:,1:2)-round(xyz_r(:,1:2))).^2,2));
%     %box_here=gather(sinc(temp));
%     %temp=xyz_r-round(xyz_r);
%     %box_here=gather((prod(sinc(temp(:,1:2)),2)));
%     %box_here=ones(size(xyz_r,1),1);%gather((prod(sinc(temp(:,1:2)),2)));
%     xyz_r=round(xyz_r)+cp;
%     inds=find(xyz_r(:,1) > 0 & xyz_r(:,2)>0 & xyz_r(:,3)>0 & ...
%     xyz_r(:,1)<=Npix & xyz_r(:,2)<=Npix & xyz_r(:,3)<=Npix);
%     xyz_r=xyz_r(inds,:);
%     %box_here=box_here(inds);
%     inds_w=sub2ind(padSize.*[1 1 1],xyz_r(:,2),xyz_r(:,1),xyz_r(:,3));    
%     frame=p_F(:,:,i);
%     %rr_here=rr(inds_w); %newVals=((rr_here(:).^2).*frame(:).*(box_here(:).^2));
%     newVals=(frame(inds));%./(box_here(:)));
%     dummy_F(inds_w)=dummy_F(inds_w)+newVals;


% % % diagnostic (test interpolation quality by rotating back):
%     dummy_r=zeros(Npix,Npix);
%     dummy_i=zeros(Npix,Npix);
%     itu=sub2ind([Npix Npix],xyz(inds,2)+cp,xyz(inds,1)+cp);
%     dummy_r(itu)=real(dummy_F(inds_w));
%     dummy_i(itu)=imag(dummy_F(inds_w));
%     dummy=complex(dummy_r,dummy_i);
%     
%     fsc=smap.radialmeanj(real((dummy.*conj(frame))./sqrt((abs(dummy).^2).*(abs(frame).^2))),[],[],'circ');
%     figure(303); hold on;
%     plot(fsc);
%     grid on; ylim([0 1.05]);
%     dummy_sinc=dummy;
%     drawnow;
%     %test=cat(3,abs(dummy_lin),abs(frame),abs(dummy_sinc));
%     
%     %figure(404); clf;
%     %imsci(abs(frame)-abs(dummy_sinc));
%     %imsci(abs(dummy_sinc));
%     %imsci(max(abs(dummy_F),[],3));
    
%    %%
    
%     
%     temp_r=interpn(dummyX,dummyX,dummyX,real(dummy_F),xyz_r(:,2),xyz_r(:,1),xyz_r(:,3));
%     temp_i=interpn(dummyX,dummyX,dummyX,imag(dummy_F),xyz_r(:,2),xyz_r(:,1),xyz_r(:,3));
%     oldVals=gather(complex(temp_r,temp_i));
%     dummy_r=zeros(Npix,Npix);
%     dummy_i=zeros(Npix,Npix);
%     itu=sub2ind([Npix Npix],xyz(inds,2)+cp,xyz(inds,1)+cp);
%     dummy_r(itu)=real(oldVals);
%     dummy_i(itu)=imag(oldVals);
%     dummy=complex(dummy_r,dummy_i);
%     clf;
%     fsc=smap.radialmeanj((abs(dummy)),[],[],'circ')./smap.radialmeanj((abs(p_F(:,:,i))),[],[],'circ');
%     plot(fsc);
%     
%  %%
%    pause;
    
% % % sinc:
% 
%     temp=sqrt(sum((xyz_r-round(xyz_r)).^2,2));
%     box_here=gather(sinc(temp));
%     xyz_r=round(xyz_r)+cp;
%     inds=find(xyz_r(:,1) > 0 & xyz_r(:,2)>0 & xyz_r(:,3)>0 & ...
%         xyz_r(:,1)<=Npix & xyz_r(:,2)<=Npix & xyz_r(:,3)<=Npix);
%     xyz_r=xyz_r(inds,:);
%     box_here=box_here(inds);
%     inds_w=sub2ind(padSize.*[1 1 1],xyz_r(:,2),xyz_r(:,1),xyz_r(:,3));    
%     frame=p_F(:,:,i);
%     %rr_here=rr(inds_w); %newVals=((rr_here(:).^2).*frame(:).*(box_here(:).^2));
%     newVals=(frame(inds).*(box_here(:)).^2);
%     dummy_F(inds_w)=dummy_F(inds_w)+newVals;
    
    
    if( ~isempty(CTF) )
        temp=(CTF(:,:,i));
        box(:,:,cp_box)=CTF(:,:,i).^2;
        if( T>0 )
            newWeights=interpn(1:Npix,1:Npix,-T:T,(box),Y,X,Z);
        else
            newWeights=interpn(1:Npix,1:Npix,(box),Y,X);
        end;
        newWeights(isnan(newWeights(:)))=0;
        W(inds_w)=W(inds_w)+gather(newWeights);
        WW(inds_w)=WW(inds_w)+ones(length(gather(inds_w)),1);
        
        %dummy_F(inds_w)=dummy_F(inds_w)+gather(newVals);
        %newVals=interpn(dummyX,dummyX,temp,Y,X);
        %newVals=temp(inds);
        %newVals(isnan(newVals(:)))=0;
        %newWeights=(rr_here(:).*box_here(:).*temp(:)).^2; % b-factor
        %newWeights=(box_here(:).*temp(inds)).^2; % sinc kernel and CTF^2
        %newWeights=(temp(inds)).^2; % CTF^2 only
        %newWeights=(box_here(:)).^2; % sinc kernel only
        
%         frame_fPSD=fPSD(:,:,i);
%         newVals_fPSD=(frame_fPSD(inds).*(box_here(:).^2));
%         otherref(inds_w)=newVals_fPSD;
        
    else
        newWeights=ones(length(inds_w),1);
        W(inds_w)=W(inds_w)+1;
    end;
    
    f_total=f_total+sum(newWeights);
    pix_total=pix_total+length(newWeights);
    
    if( mod(i,100)==0 )
        fprintf('%i\n',i);
    end;
end;
toc/i
%f=(1e-4).*(f_total./pix_total)
f=(1e-1).*(f_total./pix_total)

% %%
dummy_noW=gather(W);
% dummy_noWW=gather(WW);
% thr=1;
% inds_dummy=find(dummy_noWW(:)>thr);
% 
% f=0.1.*mean(dummy_noW(inds_dummy)) % from grigorieff 2007


% temp=gather(dummy_F);
% temp(inds_dummy)=temp(inds_dummy)./(f+dummy_noW(inds_dummy));
% outref=gather(smap.iftj(temp));
% imsc(outref(:,:,cp));

% %%
outref=gather(dummy_F);
dummy_noW=gather(W);

%pause;

% %%
% % 
% % %pause
% % %f=0.1.*mean(W(find(W(:)>0))) % from grigorieff 2007
% % % test=smap.iftj(dummy_F./(f+W));
% % % imsc(sum(test,3));
% % 
% 
% f=(1e-5).*(f_total./pix_total);
% outref=smap.iftj(dummy_F./(f+W));
% %imsc(test(:,:,290));
% imsc(sum(outref,3));


%%
if 0

    
%%
% % noise-free templates:
% CTF-multiplied: 841 (5425/6.45)
% CTF-multiplied, sinc-kernel div.: 1299 (6187/4.77)
% CTF-multiplied, (sinc-kernel * CTF^2) div.: numerical problems
% CTF-multiplied, (sinc-kernel * CTF^2) wiener filtered: 93.44 (9207/98.5)
% 
% with whitening (radial av.):
% CTF-multiplied: 4446 (5845/1.32)
% CTF-multiplied, (sinc-kernel * CTF^2) div.: <did not test>
% CTF-multiplied, sinc-kernel div.: 227 (146/0.64)
% CTF-multiplied, (sinc-kernel * CTF^2) wiener filtered: 5589 (7956/1.42)
% 
% % from expt'l images:
% % % 
% phase-flipped: 16 (291/17.84)
% CTF-multiplied: 191 (245/1.28); noenv: 190 (220/1.16)
% CTF-multiplied, sinc-kernel div.: 181 (98/0.54)
% CTF-multiplied, (sinc-kernel * CTF^2) div.: numerical problems
% CTF-multiplied, (sinc-kernel * CTF^2) wiener filtered: 60 (97.2/1.45); noenv: 68 (95/1.4)
% 
% with whitening (radial av.):
% phase-flipped: 169 (131/0.78); noenv: 169 (131/0.78)
% CTF-multiplied: 36 (21.4/0.59)
% CTF-multiplied, sinc-kernel div.: 7 (4.3/0.61)
% CTF-multiplied, (sinc-kernel * CTF^2) div.: numerical problems
% CTF-multiplied, (sinc-kernel * CTF^2) wiener filtered: 169 (114/0.67); noenv: 169 (114/0.67)

% % linear interpolation:
% % from expt'l images:
% phase-flipped: 23 (451/19.2)
% CTF-multiplied: 290 (341/1.18), 0.07ac: 151
% CTF-multiplied, (CTF^2) wiener filtered: 98 (123/1.25), 0.07ac: 77 (41/0.53)
% 
% with whitening (radial av.):
% phase-flipped: 263 (206/0.78), 0.07ac: 263 (206/0.78)
% CTF-multiplied: 65 (40/0.61); 0.07ac: 231 (166/0.7)
% CTF-multiplied, (CTF^2) wiener filtered 0.05: 248 (162/0.65)

% %%

bp_F_ref=bp_F;
SPV_F_ref=SPV_F;

Npix=size(bp_F_ref,1);
cp=floor(Npix./2)+1;

inds_zeroPix=find(abs(bp_F_ref(:))==0);
inds_goodPix=find(abs(bp_F_ref(:))>0);
goodPix=(Npix.^3)-length(inds_zeroPix);

bp_F_norm=norm3D(bp_F_ref);

%structref=structref.*otherref;
%structref(inds_goodPix)=structref(inds_goodPix)./(W(inds_goodPix));
%structref(inds_goodPix)=structref(inds_goodPix)./(W(inds_goodPix)+0.0005);
%structref=structref./(W+0.05);
%test=smap.iftj(structref);
%imsc(test(:,:,cp));
% %%
%structref(inds_goodPix)=structref(inds_goodPix)./(W(inds_goodPix)+2.23e-2);

SPV_F_z=SPV_F_ref;
SPV_F_z(inds_zeroPix)=0;
SPV_F_norm=norm3D(SPV_F_z);

cc_F=bp_F_norm.*conj(SPV_F_norm);
cc=real(fftshift(ifftn(ifftshift(cc_F))).*(sqrt(goodPix)));
fprintf('CC: %7.2f\t [%5.1f / %4.2f]\n',max(cc(:))./std(cc(:)),max(cc(:)),std(cc(:)));


SPV_F_fl=smap.ftj(flip(smap.iftj(SPV_F_ref),3));
SPV_F_fl(inds_zeroPix)=0;
SPV_F_norm=norm3D(SPV_F_z);

cc_F=bp_F_norm.*conj(SPV_F_norm);
cc=real(fftshift(ifftn(ifftshift(cc_F))).*(sqrt(goodPix)));
fprintf('CC (flipped): %7.2f\t [%5.1f / %4.2f]\n',max(cc(:))./std(cc(:)),max(cc(:)),std(cc(:)));


fPSD=smap.psdFilter_3d(abs(bp_F_ref));
bp_F_ref=bp_F_ref.*fPSD;
bp_F_norm=norm3D(bp_F_ref);

cc_F=bp_F_norm.*conj(SPV_F_norm);
cc=real(fftshift(ifftn(ifftshift(cc_F))).*(sqrt(goodPix)));
fprintf('CC: %7.2f\t [%5.1f / %4.2f]\n',max(cc(:))./std(cc(:)),max(cc(:)),std(cc(:)));



% SPV_CTF_F=SPV_F.*W;
% templateref=smap.iftj(SPV_CTF_F);
% imref=bp;
% imref=imref-mean(imref(:));
% templateref=templateref-mean(templateref(:));
% TR=templateref;
% IN=nm(imref);
% TN=nm(TR);
% sf=nansum(IN(:).*TN(:))./nansum(IN(:).*IN(:))
% bg_sub=IN-sf.*TN;
% structref=smap.ftj(bg_sub);




%test_bp=smap.iftj(bp_F_norm);
%test_SPV=smap.iftj(SPV_F_norm);

% imsc(cc(:,:,cp));

%SPV_F_norm=gpuArray(SPV_F_norm);
%test=smap.rotate3dMatrix(gpuArray(SPV_F_norm),eye(3));


%% pixel count-only SD normalization (half-volumes):

im=smap.nm(poissrnd(tt(:,:,57).*10));
im_F=smap.ftj(im);
tt_F=smap.ftj(smap.nm(tt(:,:,57)));
Npix=size(im,1);
cp=floor(Npix./2)+1;
k_2d=smap.rrj(ones(Npix,Npix));
k_2d(:,cp:end)=1e3;
mask=ones(Npix,Npix);
mask(:,cp:end)=0;
dummy=ones(Npix,Npix);
dummy=dummy.*mask;

k_filt=fliplr(linspace(0.005,0.7,100));

peaks=[]; k_val=[]; SD=[]; goodPix=[];
for i=1:length(k_filt)    
    filt_temp=single(k_2d<=k_filt(i));
    filt_temp=filt_temp(:,1:(cp-1));
    goodPix(i)=length(find(filt_temp(:)>0));
    inds=randperm((Npix.^2)./2,((Npix.^2)./2)-goodPix(i));
    filt_here=dummy; filt_here(inds)=0;

    % % filter:
    im_F_filt=im_F.*filt_here;
    tt_F_filt=tt_F.*filt_here;
    
    % % re-normalize:
    im_F_filt=im_F_filt./std(im_F_filt(:));
    tt_F_filt=tt_F_filt./std(tt_F_filt(:));    
    
    % % xcorr:
    cc_F=im_F_filt.*conj(tt_F_filt);
    cc=real(fftshift(ifftn(ifftshift(cc_F))).*(sqrt(2.*goodPix(i))));
    
    peaks(i)=cc(cp,cp);
    SD(i)=std(cc(:));
    %figure(1); imsc(cc); drawnow; pause(0.05);
    %figure(1); clf; smap.plotSH(cc); drawnow; pause(0.05);
    
end;

figure(2); clf;
subplot(2,1,1);
plot(goodPix./((Npix.^2)),(peaks).^2); hold on;
ylabel('CC^2');%, CC/\sigma')
subplot(2,1,2);
plot(goodPix./((Npix.^2)),(SD));
xlabel('fraction of pixels that are nonzero');
ylabel('\sigma');

%% pixel count-only SD normalization (full volumes):

im=smap.nm(poissrnd(tt(:,:,57).*10));
im_F=smap.ftj(im);
tt_F=smap.ftj(smap.nm(tt(:,:,57)));
Npix=size(im,1);
cp=floor(Npix./2)+1;
k_2d=smap.rrj(ones(Npix,Npix));
mask=ones(Npix,Npix);
dummy=ones(Npix,Npix);
dummy=dummy.*mask;
cp_lin=sub2ind([Npix Npix],cp,cp);
% %%
k_filt=fliplr(linspace(0.005,0.7,100));

peaks=[]; k_val=[]; SD=[]; goodPix=[];
%for i=1:50%
for i=1:length(k_filt)
    filt_temp=single(k_2d<=k_filt(i));
    filt_temp=filt_temp(1:cp_lin);
    goodPix(i)=length(find(filt_temp(:)>0));
    inds=randperm((cp_lin),((cp_lin))-goodPix(i));
    filt_here=dummy;
    filt_here(inds)=0;
    filt_here=filt_here.*rot90j(filt_here,2);

    im_F_filt=im_F.*filt_here;
    tt_F_filt=tt_F.*filt_here;

    im_F_filt=im_F_filt./std(im_F_filt(:));
    tt_F_filt=tt_F_filt./std(tt_F_filt(:));    

    cc_F=im_F_filt.*conj(tt_F_filt);
    cc=fftshift(ifftn(ifftshift(cc_F)));
    cc=real(cc.*((sqrt(2.*goodPix(i)))));

    peaks(i)=cc(cp,cp);
    SD(i)=std(cc(:));
    figure(1); clf; smap.plotSH(cc); drawnow; pause(0.05);
end;

figure(2); clf;
subplot(2,1,1);
plot(goodPix./((Npix.^2)),peaks.^2); hold on;
%plot(goodPix./((Npix.^2)),peaks./SD,'r');
ylabel('CC');%, CC / \sigma')
subplot(2,1,2);
plot(goodPix./((Npix.^2)),(SD));
xlabel('fraction of pixels that are nonzero');
ylabel('\sigma');

%% pixel count-only SD normalization (half-volumes; with extra lines):

im=smap.nm(poissrnd(tt(:,:,57).*10));
im_F=smap.ftj(im);
tt_F=smap.ftj(smap.nm(tt(:,:,57)));
Npix=size(im,1);
cp=floor(Npix./2)+1;
k_2d=smap.rrj(ones(Npix,Npix));
k_2d(:,cp:end)=1e3;
mask=ones(Npix,Npix);
mask(:,cp:end)=0;
dummy=ones(Npix,Npix);
dummy=dummy.*mask;
% %%
k_filt=fliplr(linspace(0.005,0.7,100));

peaks=[]; k_val=[]; SD=[]; goodPix=[];
%for i=1:1%50%
for i=1:length(k_filt)    
    filt_temp=single(k_2d<=k_filt(i));
    filt_temp=filt_temp(:,1:(cp-1));
    goodPix(i)=length(find(filt_temp(:)>0));
    inds=randperm((Npix.^2)./2,((Npix.^2)./2)-goodPix(i));
%     inds_rev=setdiff(1:(Npix.^2)/2,inds);
    filt_here=dummy;
    filt_here(inds)=0;
    im_F_filt=im_F.*filt_here;
    tt_F_filt=tt_F.*filt_here;
    
%     im_F_filt=im_F_filt./std(im_F_filt(inds_rev));
%     tt_F_filt=tt_F_filt./std(tt_F_filt(inds_rev));
%     cc_F=im_F_filt.*conj(tt_F_filt);
%     SD(i)=std(cc_F(:));
%     %cc=real(fftshift(ifftn(ifftshift(cc_F))).*((Npix.^2)./(2.*sqrt(goodPix(i)))));
%     cc=real(fftshift(ifftn(ifftshift(cc_F))).*((Npix.^2).*sqrt(2)./(sqrt(goodPix(i)))));
    
    im_F_filt=im_F_filt./std(im_F_filt(:));
    tt_F_filt=tt_F_filt./std(tt_F_filt(:));    
    cc_F=im_F_filt.*conj(tt_F_filt);
    cc=real(fftshift(ifftn(ifftshift(cc_F))).*(sqrt(2.*goodPix(i))));
    
    peaks(i)=cc(cp,cp);
    SD(i)=std(cc(:));
    %figure(1); imsc(cc); drawnow; pause(0.05);
end;
figure(2); clf;
subplot(2,1,1);
plot(goodPix./((Npix.^2)),(peaks).^2); hold on;
%plot(goodPix./((Npix.^2)),(peaks./SD).^2,'r');
ylabel('CC^2');%, CC/\sigma')
subplot(2,1,2);
plot(goodPix./((Npix.^2)),(SD));
xlabel('fraction of pixels that are nonzero');
ylabel('\sigma');

%%

% zz=single(z_cut(:,:,7:36));
% zz_cut=smap.cutj(zz,[384,384,30]);
% zz_cut=smap.nm(zz_cut);
% zz_cut=smap.extendj(zz_cut,[384,384,384],0);
% rec=smap.backproject(zz_cut,Rref);
% rec=gather(rec);
z_r=[];

for i=1:size(z,3)
    temp=z(:,:,i);
    temp=smap.extendj(temp,2048.*[1,1],mode(temp(:)));
    z_r(:,:,i)=smap.cutj(smap.rotate2dMatrix(temp,RotationMatrix(quaternion.eulerangles('zyz',[12 0 0].*pi./180))),[size(z,1),size(z,2)]);
end;
%z=z_r;
%%
binVal=2;
zBin=4/binVal;

tilts=(-42:3:45)';
%tilts=(-45:3:42)';
%tilts=(-60:3:60)';
%tilts=(60:-3:-60)';
%tilts=tilts-30;
nTilts=length(tilts);
angles=[zeros(nTilts,1) tilts zeros(nTilts,1)];
Rref=normalizeRM(squeeze(RotationMatrix(quaternion.eulerangles('xyz', ...
    angles.*pi./180))));

R_offset=normalizeRM(RotationMatrix(quaternion.eulerangles('xyz',[0 0 0].*pi./180)));
for i=1:size(Rref,3)
    Rref(:,:,i)=R_offset*Rref(:,:,i);
end;
Rref=normalizeRM(Rref);

%zz=z;
zz=z_r;
%zz=tt;
nFrames=size(zz,3);
Npix=size(zz,1);
%Npix=min([size(zz,1) 784]);
zz_cut=smap.cutj(zz,[Npix,Npix,nFrames]);
zz_test=zeros(Npix/binVal,Npix/binVal,size(zz_cut,3),'single');
if( binVal>1 )
    for i=1:nFrames
        zz_test(:,:,i)=(smap.resize_F(zz_cut(:,:,i),1/binVal,'newSize'));
    end;
    
else
    zz_test=zz_cut;
end;

rec=smap.backproject(zz_test,Rref,size(zz_test,1).*2);
%rec=gather(smap.iftj(rec));
cp=floor(size(rec,3)./2)+1;


% %%
nSections=size(rec,3)./zBin;
rec_gz=[];
for i=1:nSections
    inds=[((i-1)*zBin+1):(min([i*zBin size(rec,3)]))];
    rec_gz(:,:,i)=sum(rec(:,:,inds),3);
end;
cp=floor(size(rec_gz,3)./2)+1;
%imsc(rec_gz(:,:,cp));
rec_gz=rec_gz-min(rec_gz(:));
rec_gz_log=real(log(rec_gz));

imscj(rec_gz);

% rec_rot=smap.rotate3dMatrix(gpuArray(rec),RotationMatrix(quaternion.eulerangles('xyz',[-20 0 0].*pi./180)));
% rec_rot=gather(smap.iftj(rec_rot));
% imsc(squeeze(sum(rec_rot,2)))
% nSections=size(rec_rot,3)./zBin;
% rec_gz=[];
% for i=1:nSections
%     inds=[((i-1)*zBin+1):(min([i*zBin size(rec_rot,3)]))];
%     rec_gz(:,:,i)=sum(rec_rot(:,:,inds),3);
% end;
% cp=floor(size(rec_gz,3)./2)+1;

%%

binVal=2;
zBin=4;

tilts=(-42:3:45)';
nTilts=length(tilts);
angles=[zeros(nTilts,1) tilts zeros(nTilts,1)];
Rref=normalizeRM(squeeze(RotationMatrix(quaternion.eulerangles('xyz', ...
    angles.*pi./180))));

R_offset=normalizeRM(RotationMatrix(quaternion.eulerangles('xyz',[0 0 0].*pi./180)));
for i=1:size(Rref,3)
    Rref(:,:,i)=R_offset*Rref(:,:,i);
end;
Rref=normalizeRM(Rref);

zz=single(z);

nFrames=size(zz,3);
Npix=size(zz,1);
%Npix=min([size(zz,1) 784]);
zz_cut=smap.cutj(zz,[Npix,Npix,nFrames]);
Npix_bin=Npix/binVal;

%%
edgeSmall=128;
padSize=256;
temp=smap.rrj(ones(edgeSmall)).*edgeSmall;
mask=single(temp<=10);

zz_test=zeros(Npix/binVal,Npix/binVal,size(zz_cut,3),'single');
if( binVal>1 )
    for i=1:nFrames
        zz_test(:,:,i)=(smap.resize_F(zz_cut(:,:,i),1/binVal,'newSize'));
    end;
end;

for j=1:nFrames
    %inds_i=setdiff([(j-2):(j+2)],j);%
    inds_i=setdiff(1:nFrames,j);
    inds_e=j;%setdiff(1:size(z,3),inds_i);
    Rref_i=Rref(:,:,inds_i);
    Rref_e=Rref(:,:,inds_e);
    zz_i=(zz_test(:,:,inds_i));
    zz_e=sum(zz_test(:,:,inds_e),3);
    
    zz_e=smap.cutj(zz_e,edgeSmall.*[1,1]);
    zz_i=smap.cutj(zz_i,[edgeSmall,edgeSmall,length(inds_i)]);
    
    
    rec_i=smap.backproject(zz_i,Rref_i,padSize);
    cp=floor(size(rec_i,3)./2)+1;
    temp=smap.iftj(smap.rotate3dMatrix(smap.ftj(gpuArray(rec_i)),Rref_e));
    temp_s=gather(sum(temp(:,:,(cp-10):(cp+10)),3));
    
    cc=smap.ccf(temp_s,zz_e);%.*mask;
    [nx,ny]=find(cc==max(cc(:)),1,'first');
    [nx-cp ny-cp]
    imsc(cc); drawnow;
    
    zz_e=sum(zz_test(:,:,inds_e),3);
    temp=circshift(zz_e,[nx-cp,ny-cp]);
    zz_test(:,:,j)=temp;
end;

rec_s=smap.backproject(zz_test,Rref,size(zz_test,1).*2);

%%

%angles=[zeros(30,1) (-42:3:45)' zeros(30,1)];
angles=[zeros(30,1) (-42:3:45)' zeros(30,1)];

Rref=normalizeRM(squeeze(RotationMatrix(quaternion.eulerangles('xyz', ...
    angles.*pi./180))));

R_offset=normalizeRM(RotationMatrix(quaternion.eulerangles('zyz',[15 0 0].*pi./180)));
for i=1:size(Rref,3)
    Rref(:,:,i)=R_offset*Rref(:,:,i);
end;
Rref=normalizeRM(Rref);

Rref=normalizeRM(squeeze(RotationMatrix(quaternion.randRot(1,300))));
tt=smap.templates_gpu(s,Rref,[]);
ttt=[];
for i=1:size(tt,3)
    ttt(:,:,i)=smap.iftj(tt(:,:,i));
end;
tt=ttt;
clear ttt;

%% old version (with 3D rotations)

dummy=zeros(Npix,Npix,Npix,'single');
dummy=dummy+1i*dummy;
dummy_F=dummy;
dummy=gpuArray(dummy);
%dummy_F_c=dummy;
W=ones(Npix,Npix,Npix,'single');

%pause;

% [y y] xx
% [n y] ~
% [y n] xx
% [n n] ~
fprintf('Adding slices...\n');
for i=1:nPatches
    Rref_here=Rref(:,:,i);
    RM=Rref_here';%Rref(:,:,i)';
    dummy(:,:,cp)=p_F(:,:,i);
    dummy(cp,cp,cp)=0;
    dummy_r=smap.rotate3dMatrix(dummy,RM);
    dummy_F=dummy_F+gather(dummy_r);

    %RM=Rref_here;
    xyz_r=(RM*xyz')';
    xyz_r=round(xyz_r)+cp;
    xyz_r(xyz_r>Npix)=Npix;
    xyz_r(xyz_r<1)=1;
    %inds_w=sub2ind(padSize.*[1 1 1],xyz_r(:,1),xyz_r(:,2),xyz_r(:,3));
    inds_w=sub2ind(padSize.*[1 1 1],xyz_r(:,2),xyz_r(:,1),xyz_r(:,3));
    W(inds_w)=W(inds_w)+1;
     
    if( mod(i,10)==0 )
        fprintf('%i\n',i);
    end;
end;
%pause

dummy_F=dummy_F./W;



%%
nPatches=50;
Npix=64;
Rtu=RR(:,:,1:50);

p=pbf{2}; p(isnan(p))=0;
p_F=zeros(5,5,nPatches);
p_F=p_F+1i*p_F;
p_c=pbf_c{2}; p_c(isnan(p_c))=0;
p_F_c=zeros(5,5,nPatches);
p_F_c=p_F_c+1i*p_F_c;
for i=1:nPatches
    p_F(:,:,i)=smap.ftj(p(:,:,i));
    p_F_c(:,:,i)=smap.ftj(p_c(:,:,i));
end;

p_F=smap.extendj(p_F,[Npix,Npix,nPatches],0);
p_F_c=smap.extendj(p_F_c,[Npix,Npix,nPatches],0);

cp=floor(Npix./2)+1;
[x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
x0=x(:,:,cp)-cp;
y0=y(:,:,cp)-cp;
z0=zeros(Npix,Npix);
xyz=double([x0(:) y0(:) z0(:)]);

dummy=zeros(Npix,Npix,Npix,'single');
dummy=dummy+1i*dummy;
dummy_F=dummy;
dummy_F_c=dummy;
W=ones(Npix,Npix,Npix,'single');

dummyX=1:size(dummy,1); dummyY=1:size(dummy,2); dummyZ=1:size(dummy,3);
clear x y z x0 y0 z0;

randInds=randperm(nPatches);
for i=2:nPatches
    %RM=Rtu(:,:,randInds(i))';
    RM=Rtu(:,:,i);
    %RM=RM'; % should be the inverse
    dummy(:,:,cp)=p_F(:,:,i);
    dummy(cp,cp,cp)=0;
    dummy_r=smap.rotate3dMatrix(dummy,RM);
    dummy_F=dummy_F+dummy_r;
    
    dummy(:,:,cp)=p_F_c(:,:,i);
    dummy(cp,cp,cp)=0;
    dummy_r=smap.rotate3dMatrix(dummy,RM);
    dummy_F_c=dummy_F_c+dummy_r;

    xyz_r=(RM*xyz')';
    xyz_r=round(xyz_r)+cp;
    xyz_r(xyz_r>Npix)=Npix;
    xyz_r(xyz_r<1)=1;
    inds_w=sub2ind([64 64 64],xyz_r(:,1),xyz_r(:,2),xyz_r(:,3));
    W(inds_w)=W(inds_w)+1;%sub2ind([64 64 64],xyz_r(1,:))
    %output_image = interp3gpu(dummyX,dummyY,dummyZ,V_Fr,V_Fi,xyz_r(:,2)+cp,xyz_r(:,1)+cp,xyz_r(:,3)+cp); wait(gdev); % ~4 ms
    %projPot=reshape(output_image,Npix,Npix);
    %projPot=real(ifftn(projPot(idx_small_i{:})));
end;

dummy_F=dummy_F./W;
dummy_F_c=dummy_F_c./W;

dz=smap.iftj(dummy_F);
[nx,ny,nz]=ind2sub([64 64 64],find(dz==max(dz(:))));
disp([nx ny nz]);

dz_c=smap.iftj(dummy_F_c);
[nx_c,ny_c,nz_c]=ind2sub([64 64 64],find(dz_c==max(dz_c(:))));
disp([nx_c ny_c nz_c]);


%% test case:

rr=smap.rrj(ones(64,64,64)).*64;
z=g2(rr,[1 2]);
z_s=circshift(z,[3 5 7]);
z_s=smap.cutj(z_s,32.*[1,1,1]);

z=smap.mr(s.prop.SPName);
zz=circshift(z,[5,10,15]);
zzz=smap.cutj(zz,128.*[1,1,1]);
z_s=zzz;

%%
seedWords={'mt19937ar','Seed',eval('1')};
stream = RandStream(seedWords{1},seedWords{2},seedWords{3});
RandStream.setGlobalStream(stream);

nRotations=1000;
R_test=normalizeRM(squeeze(RotationMatrix(quaternion.randRot(1,nRotations))));

z_s_F=smap.ftj(gpuArray(z_s));
z_proj=[];
for i=1:nRotations
    z_proj(:,:,i)=gather(sum(smap.iftj(smap.rotate3dMatrix(z_s_F,R_test(:,:,i))),3));
    if( mod(i,100)==0 )
        fprintf('%i\n',i);
    end;
end;


%%

padSize=128;
F=zeros(padSize,padSize,padSize,'single');
W=F;
for i=1:nRotations
     patchref=z_proj(:,:,i);
     [dummy_F,dummy_W]=smap.backproject(patchref,R_test(:,:,i),padSize);
     F=F+dummy_F;
     W=W+dummy_W;
     if( mod(i,100)==0 )
         fprintf('%i\n',i);
         imsc(sum(smap.iftj(F./W),3)); title(i); drawnow;
     end;
end;

rec=smap.iftj(F./W);
%rec=smap.iftj(F);
imsc(sum(rec,3));








%%








%%
end;

