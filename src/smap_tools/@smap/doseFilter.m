function outref=doseFilter(imref,totalDose,aPerPix,varargin);
% % all we are trying to do here is match the ks of the resulting
% summed image to the expected ks given this model for radiation
% damage. So just use the model to generate a filter for dose- and k-dependent amplitude
% attenuation. Optimal SNR in a 3D reconstruction is not sought. Here,
% the matched filter is the template, and whitening should pick up on
% this dose-dependent loss in high ks.
%
% function outref=doseFilter(imref,totalDose,aPerPix);


%disp('paused'); pause;

norm_type='numerator_only';
condition='LN';
if( nargin>3 )
    norm_type=varargin{1};
    if( nargin>4 )
        condition=varargin{2};
    end;
end;

edgeSize=size(imref,1);
nFrames=size(imref,3);
dosePerFrame=totalDose./nFrames;

k=single(smap.getKs(imref(:,:,1),aPerPix));
%k_1d=smap.radialmeanj(k,[],[],'circ');
inds=find(abs(k(:)-0.125)<0.005);
%SNR=zeros(nFrames,length(k_1d));

%pause;

% %%
a=0.24499;%0.245;
b=-1.6649;%-1.665;
c=2.8141;%2.81;

Nc=a.*k.^b+c;
%Nc_1d=a.*k_1d.^b+c;

switch condition
    case 'LN'

    case 'LHe'
        Nc=Nc.*2;
        Nc_1d=Nc_1d.*2;        
end;

Nopt=2.51284.*Nc;
%Nopt_1d=2.51284.*Nc_1d;

outref=zeros(edgeSize,edgeSize,'single');
q=zeros(size(imref,1),size(imref,2));
q2=q;
q0=ones(size(imref,1),size(imref,2));

q_sample=[];
q_pre_sample=[];
q_all=zeros(size(imref,1),size(imref,2));
for i=1:nFrames
    N=dosePerFrame.*i; 
    %SNR(i,:)=(((1-exp(-N./(2.*Nc_1d))).^2)./N);%.*(N<Nopt_1d);
    
    q=exp(-N./(2.*Nc));
    q_all=q_all+q;
    q_pre_sample(i)=mean(q(inds));    
    %q=q0;
    %q=q.*(N<Nopt);

    q2=q2+q.^2;
    qq=q;
    q_sample(i)=mean(q(inds));
%    temp=gpuArray(imref(:,:,i));
    temp=imref(:,:,i);
    dcVal=mean(temp(:));
    temp=temp-dcVal;
    temp=smap.iftj(smap.ftj(temp).*q);
    outref=outref+gather(temp+dcVal);
    if( mod(i,10)==0 )
        %fprintf('%s\n',num2str(i));
    end;

end;
%outref=outref-mean(outref(:));
if( strcmp(norm_type,'noise_restored') )
    outref=smap.iftj(smap.ftj(outref)./sqrt(q2));
elseif( strcmp(norm_type,'numerator_only') )
    %fprintf('using filter with numerator only...\n');
end;


%outref=gather(outref);




if( 0 )

%%
ftu=[10:40];
mv_filt_new=nan(length(ftu),94);
mv_unfilt_new=nan(length(ftu),94);
for j=1:length(ftu)
    nFrames=ftu(j);
    mov_filt_new=smap.doseFilter(mov(:,:,1:nFrames),nFrames.*0.5,0.97);
    cc_filt_new=smap.ccff(sum(mov_filt_new,3),tt_n);
    mov_unfilt_new=mov(:,:,1:nFrames);
    cc_unfilt_new=smap.ccff(sum(mov_unfilt_new,3),tt_n);
    for i=1:94
        temp=cc_unfilt_new(:,:,i);
        mv_unfilt_new(j,i)=max(temp(:));
        temp=cc_filt_new(:,:,i);
        mv_filt_new(j,i)=max(temp(:));
    end;
    nanmean(mv_filt_new-mv_unfilt_new,2)
end;

ftu=[10:10:100];
mv_filt_new=nan(length(ftu),size(tt_n,3));
mv_unfilt_new=nan(length(ftu),size(tt_n,3));
for j=1:length(ftu)
    nFrames=ftu(j);
    mov_filt_new=smap.doseFilter(mov_shifted(:,:,1:nFrames),nFrames.*0.35,1.032);
    cc_filt_new=smap.ccff(sum(mov_filt_new,3),tt_n);
    %mov_unfilt_new=mov(:,:,1:nFrames);
    %cc_unfilt_new=smap.ccff(sum(mov_unfilt_new,3),tt_n);
    for i=1:size(tt_n,3)
        %temp=cc_unfilt_new(:,:,i);
        %mv_unfilt_new(j,i)=max(temp(:));
        temp=cc_filt_new(:,:,i);
        mv_filt_new(j,i)=max(temp(:));
    end;
    %nanmean(mv_filt_new-mv_unfilt_new,2)
    plot(mv_filt_new); drawnow;
end;    
    
    
    
    
    
    
    
    
    
    
    
    
end;
