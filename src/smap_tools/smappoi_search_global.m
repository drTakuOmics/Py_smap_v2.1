function varargout=smappoi_search_global(paramsFile,jobref);
% compile with:
% /misc/local/matlab-2023b/bin/mcc -mv -I /groups/denk/home/rickgauerj/smap_v2.1/release/src/smap_tools/ -I /groups/denk/home/rickgauerj/smap_v2.1/release/src/emClarity_FFT/ -R -singleCompThread /groups/denk/home/rickgauerj/smap_v2.1/release/src/smap_tools/smappoi_search_global.m
compile_date='10.25.24';

%warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
clearvars -global 
global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global Rref
global imref
global nfIm meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new
global const inds_mask bgVal params nfIm_unmod gdev

jobNum=str2num(jobref);

[~,this_server]=system('uname -n');


%
% set up GPUs:
% gtu=mod(jobNum-1,8)+1;
words=['nvidia-smi -L'];
[~,resp]=system(words);
temp=splitlines(resp);
inds_gpus=find(~cellfun(@isempty,temp));
temp=temp(inds_gpus)
n_gpus=length(inds_gpus);
gtu=mod(jobNum-1,n_gpus)+1
gpu_name=temp{gtu}

fprintf('getting gpu # %s...',num2str(gtu));
try
    gdev=gpuDevice(gtu);
catch
    fprintf('Failed to get gpu # %s\n',num2str(gtu));
    fidFail=fopen([pwd '/fail_' smap.zp(jobNum,4) '.txt'],'w');
    fprintf(fidFail,'%s\n',datestr(now,31));
    try
        fprintf(fidFail,'%s\n',gpu_name);
    end;
    fclose(fidFail);
%     if( debug_flag )
%         gdev=gpuDevice(1);
%     else
        return;
%     end;
end;

if( ~isempty(paramsFile) )
    try
        params=smap.readParamsFile(paramsFile);
    catch
        fprintf('Problem reading parameter file %s...\n',char(paramsFile));
        return;
    end;
end;

% % for backwards compatibility (10.05.20/jpr):
if( isempty(params.structureFile) & ~isempty(params.PDBFile) )
    params.structureFile=params.PDBFile;
elseif( isempty(params.structureFile) & ~isempty(params.modelFile) )
    params.structureFile=params.modelFile;
end;

if( ~isempty(params.structureFile) )
    [~,~,ext]=fileparts(params.structureFile)
    switch ext
        case {'.cif','.pdb','.PDB','.pdb1'}
            fn_SP=smappoi_calculate_SP(params,jobNum);
            params.modelFile=fn_SP;
            wait_flag=1;
            while wait_flag
                if( exist(fn_SP,'file')>0 )
                    try
                        SPV=smap.mr(params.modelFile);
                        wait_flag=0;
                        break
                    catch
                        wait_flag=1;
                    end;
                end;
                pause(1);
            end
        case '.mrc'
            params.modelFile=params.structureFile; 
            SPV=smap.mr(params.modelFile);
        otherwise
            fprintf(fidFail,'Structure filetype not recognized for %s...exiting\n',char(params.structureFile));
    end;

    % % end insert calcSP here
end;


scratchDir=fullfile(params.outputDir,'scratch');
if( exist(params.outputDir,'dir')~=7 )
    fprintf('Making new project directory... [%s]\n',params.outputDir);
    mkdir(params.outputDir);
end;
if( exist(scratchDir,'dir')~=7 )
    fprintf('Making new scratch directory... [%s]\n',scratchDir);
    mkdir(scratchDir);
end;

disp(datestr(now));
fileNum=jobNum;
fnLog=[scratchDir '/output_' smap.zp(jobNum,4) '.txt'];
fprintf('Making new logfile... [%s]\n',fnLog);
fidLog=fopen(fnLog,'w');
fprintf(fidLog,'%s\n',datestr(now,31));
fprintf(fidLog,'Compiled %s\n',compile_date);

fileBase='search_';
fNumPadded=smap.zp(num2str(fileNum),4);

searchDir=[scratchDir '/'];
outputDir=[params.outputDir '/'];
if( ~isempty( params.angle_inc ) )
    if( isempty( params.psi_inc ) )
        params.psi_inc=params.angle_inc;
    end;
    fprintf(fidLog,'calculating search grid for angle_inc=%6.3f, psi_inc=%6.3f...\n',params.angle_inc,params.psi_inc);
    R=smap.calculate_search_grid('C1', params.angle_inc, params.psi_inc);
    params.range_degrees=(params.angle_inc+params.psi_inc)./2;
    fprintf(fidLog,'angular refinement range set to (angle_inc + psi_inc)/2 (=%6.3f)\n',params.range_degrees);
    if( jobNum==1 )
        try
            smap.writeRotationsFile(R,fullfile([outputDir '/rotations.txt']));
            fprintf(fidLog,'Wrote custom rotations file to output dir\n');
        catch
        end;
    end;
else
    fprintf(fidLog,'reading rotations file %s...\n',char(params.rotationsFile));
    R=smap.readRotationsFile(params.rotationsFile);
end;

nIndices=size(R,3);
R_inds=smap.assignJobs(nIndices,params.nCores,jobNum);
rotationsPerFile=length(R_inds);
nRotations=rotationsPerFile;
fileMultiplier=R_inds(1)-1;
fprintf(fidLog,'job indices: [%s:%s]\n',num2str(R_inds(1)),num2str(R_inds(end)));
fprintf('job indices: [%s:%s]\n',num2str(R_inds(1)),num2str(R_inds(end)));

nRotations_full=size(R,3);

ccFlag=1;
if( isempty(params.imageFile) )
    ccFlag=0;
end;

if( strcmp(params.defocus_format,'CTFFind') )
    fprintf(fidLog,'Converting defocus estimate from CTFFind format\n');
    params.defocus(1:2)=params.defocus(1:2)./10;
    params.defocus(3)=params.defocus(3).*(-pi./180); % % nb: converts from input in CTFFind convention (degrees, not radians)
end;

T_sample=params.T_sample;
df_inc=params.df_inc;
nDfs=1;
if( T_sample>0 )
    temp=[-T_sample/2:df_inc:T_sample/2];
    temp=temp-mean(temp);
    nDfs=length(temp);
    df_new=[repmat(params.defocus(1:2),nDfs,1)+repmat(temp',1,2) repmat(params.defocus(3),nDfs,1)]
    params.defocus=df_new;
end;

if( isempty(params.aPerPix_search) )
    params.aPerPix_search=params.aPerPix;
    fprintf(fidLog,'no binning for global search...\n');
end;

%
% read in the orientations to test specific to this run:
pv=zeros(nRotations,1);
pl=zeros(nRotations,1);
SD=zeros(nRotations,1);
fprintf(fidLog,'\nJob %i of %i, including %i rotations...\n',jobNum,params.nCores,nRotations);
disp(params);
aPerPix_orig=params.aPerPix;

if( ccFlag )
    % padding:
    imref_orig=smap.mr(params.imageFile);
    imref=gather(imref_orig);
    if( aPerPix_orig ~= params.aPerPix_search )
        imref=smap.resize_F(imref,aPerPix_orig./params.aPerPix_search,'newSize');
        params.aPerPix=params.aPerPix_search;
    end;
    
    dims_imref=size(imref);
    cp_imref=floor(dims_imref./2)+1
    
    imref=smap.resizeForFFT(imref,'pad',0);
    
    imref_copy=imref;
    Npix_im=size(imref);
    cp_im=floor(Npix_im./2)+1;
    
    % % verify this:
    Npix_im_norm=sqrt(prod(Npix_im));
    
    if( ~isempty(params.maskFile) )
        if( exist(params.maskFile,'file')==2 )
            mask_im=smap.mr(params.maskFile);
            Npix_mask=sqrt(length(find(mask_im(:)==1)));
            mask_im=smap.resizeForFFT(mask_im,'crop');
            fprintf(fidLog,'using mask in %s with %7.0d pixels...\n', ...
                params.maskFile,length(find(mask_im(:)==1)));
        else
            mask_im=ones(size(imref,1),size(imref,2));
            Npix_mask=size(imref,1);
            fprintf(fidLog,'could not find specified mask file %s...\n',params.maskFile);
        end;
    else
        mask_im=ones(size(imref,1),size(imref,2));
        Npix_mask=size(imref,1);
        fprintf(fidLog,'no mask specified for search image...\n');
    end;
    
    if( params.psdFilterFlag )
        try
            fprintf(fidLog,'calculating PSD filter...\n');
            imref=imref-mean(imref(:));
            [fPSD,imref]=smap.psdFilter(imref,'sqrt');
            imref=smap.nm(imref);
            fPSD=single(ifftshift(fPSD));
            fprintf(fidLog,'PSD filter calculated\n');
        catch
            fprintf(fidLog,'could not find PSD filter...\n');
            fprintf(fidLog,'not applying PSD filter...\n');
            fPSD=ones(size(imref,1),size(imref,2),'single');
            fPSD=smap.resize_F(fPSD,size(imref,1)/size(fPSD,1),'newSize');
        end;
    else
        fprintf(fidLog,'not applying PSD filter...\n');
        fPSD=ones(size(imref,1),size(imref,2),'single');
        fPSD=smap.resize_F(fPSD,size(imref,1)/size(fPSD,1),'newSize');
        fPSD=ifftshift(fPSD);
        fPSD(1,1)=0;
        fPSD=fftshift(fPSD);
    end;
    if( params.maskCrossFlag )
        imref=smap.mask_central_cross(imref);
    end;
    imref=smap.cropOrPad(imref,dims_imref);
    imref=smap.nm(imref);
    imref=smap.resizeForFFT(imref,'pad',0);
    
    imref_F=fftn(ifftshift(single(imref)))./Npix_im_norm;
    fprintf(fidLog,'imref_F normalized, quad-swapped\n');
    
    % if this is the first job, save a copy of the filtered image:
    if( fileNum==1 )
        try
            copyfile(paramsFile,outputDir);
        catch
        end;
        mw(single(imref),[outputDir 'searchImage.mrc'],params.aPerPix);
    end;
    
    %         disp(['using image with ' num2str(Npix_im_norm) '^2 image pixels, ' num2str(prod(Npix_im)) '^2 total pixels']);
    fprintf('using image with %i x %i image pixels, %i x %i total pixels\n',dims_imref(1),dims_imref(2),Npix_im(1),Npix_im(2));
    fprintf(fidLog,'using image with %i x %i image pixels, %i x %i total pixels\n',dims_imref(1),dims_imref(2),Npix_im(1),Npix_im(2));
    
end; % if( ccFlag )


% read in the scattering potential and forward transform (leave origin at center):
disp('reading SPV...');
SPV=smap.mr(params.modelFile);
fprintf(fidLog,'SPV read\n');



if( params.aPerPix_search ~= aPerPix_orig )
    SPV=smap.resize_F(SPV,aPerPix_orig./params.aPerPix_search,'newSize');
    SPV=smap.resizeForFFT(SPV,'crop');
    fprintf(fidLog,'resized\n');
end;
if( ~isempty(params.mask_params) )
    [SPV,~]=smap.mask_a_volume(SPV,params.mask_params);
    fprintf(fidLog,'masked\n');
end;

Npix=single(min(size(SPV)));
Npix_c=Npix;
% V=SPV;
V=SPV+1i*params.F_abs*SPV;
V_F=double(fftshift(fftn(ifftshift(V))));
cp=floor(size(V_F,1)./2)+1;

V_F(cp,cp,cp)=0;

V_Fr=double(real(V_F));
V_Fi=double(imag(V_F));
clear SPV V V_F;
fprintf(fidLog,'SPV prepared\n');

binRange=params.binRange;
nBins=params.nBins;
bins=linspace(-binRange,binRange,nBins);
N=zeros(nBins,1);
fprintf(fidLog,'bins ready\n');
margin_pix=params.margin_pix;
fprintf(fidLog,'margin_pix: %i\n',margin_pix);

% % work out re-indexing matrices equivalent to small and large
% % fftshifts in advance:
% fftshift:
idx_large_f = cell(1,2);
for k = 1:2
    m = Npix_im(k);
    p = ceil(m/2);
    idx_large_f{k} = single([p+1:m 1:p]);
end;
idx_small_f = cell(1,2);
for k = 1:2
    m = Npix;
    p = ceil(m/2);
    idx_small_f{k} = [p+1:m 1:p];
end;

% ifftshift:
idx_large_i = cell(1,2);
for k = 1:2
    m = Npix_im(k);
    p = floor(m/2);
    idx_large_i{k} = single([p+1:m 1:p]);
end;
idx_small_i = cell(1,2);
for k = 1:2
    m = Npix;
    p = floor(m/2);
    idx_small_i{k} = [p+1:m 1:p];
end;

fprintf(fidLog,'fftshifts found\n');

[k_2d,cp]=smap.getKs(Npix,params.aPerPix);
[x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
x0=x(:,:,cp)-cp;
y0=y(:,:,cp)-cp;
z0=zeros(Npix,Npix);
xyz=double([x0(:) y0(:) z0(:)]);
dummyX=1:size(V_Fr,1); dummyY=1:size(V_Fr,2); dummyZ=1:size(V_Fr,3);
clear x y z x0 y0 z0;

fprintf(fidLog,'dummy grids ready\n');

%%

meanDfs=[];

CTF=smap.ctf(params.defocus,[Npix_im(1) Npix_im(2)]);
fprintf(fidLog,'initial CTFs calculated\n');
for i=1:nDfs
    CTF(:,:,i)=ifftshift(CTF(:,:,i));
end;
fprintf(fidLog,'CTF quad-swapped\n');

CTF=imag(CTF);
fprintf(fidLog,'imag. part retained\n');

fnDone=[searchDir fileBase '_' fNumPadded '.dat'];
fprintf(fidLog,'fnDone is %s\n',fnDone);
R_full=R;
R=double(R(:,:,R_inds));
fprintf(fidLog,'allocated my rotations, with %i to test\n',size(R,3));
R=smap.normalizeRM(R);


%%

fprintf(fidLog,'rotations normalized\n');

if( ccFlag )
    
    % % work out the shifts for padding templates out just once:
    xDim=Npix_im(1); yDim=Npix_im(2);
    oldDim=[(Npix) (Npix)];
    newDim=[(xDim) (yDim)];
    halfOldDim=[]; halfNewDim=[]; centerPixOld=[]; centerPixNew=[];
    for i=1:2
        oddOldFlag=mod(oldDim(i),2);
        if( oddOldFlag==1 )
            halfOldDim(i)=ceil(oldDim(i)./2);
            halfNewDim(i)=floor(newDim(i)./2);
            centerPixOld(i)=halfOldDim(i);
            centerPixNew(i)=halfNewDim(i)+1;
        else
            halfOldDim(i)=oldDim(i)./2;
            halfNewDim(i)=newDim(i)./2;
            centerPixOld(i)=halfOldDim(i)+1;
            centerPixNew(i)=floor(halfNewDim(i)+1);
        end;
        edges{i}=[centerPixOld(i)-newDim(i)./2 centerPixOld(i)+(newDim(i)./2)-1];
    end;
    diffSize=newDim(1)-oldDim(1);
    rowColInds={ [num2str(centerPixNew(1)-oldDim(1)./2) ':' num2str((centerPixNew(1)+oldDim(1)./2)-1)] , ...
        [num2str(centerPixNew(2)-oldDim(2)./2) ':' num2str((centerPixNew(2)+oldDim(2)./2)-1)] };
    %         dummy=reshape(1:(xDim)*(yDim),(yDim),(xDim));
    dummy=reshape(1:(xDim)*(yDim),(xDim),(yDim));
    eval(['rowColNums=dummy(' rowColInds{1} ',' rowColInds{2} ');']);
    rowColNums=single(rowColNums);
    
    mip=ones(Npix_im(1),Npix_im(2),'single').*-1000;
    mipi=ones(Npix_im(1),Npix_im(2),'single').*-1000;
    sum1=zeros(Npix_im(1),Npix_im(2),nDfs,'double');
    sum2=zeros(Npix_im(1),Npix_im(2),nDfs,'double');
    temp_image=zeros(Npix_im(1),Npix_im(2),'single');
    
    fprintf(fidLog,'computing xcorrs...\n'); tic;
else
    fprintf(fidLog,'making templates (no image to search)...\n');
    templates=zeros(Npix,Npix,nRotations,'single');
    tic;
end; % if( ccFlag )


% % initialize variables:
qInd=1; i=1; bgVal=[]; X=[]; Y=[]; Z=[]; output_image=[];
vi_rs=[]; projPot=[]; ew=[]; w_det=[]; template=[]; temp=[];
rMask=[]; bgVal=[]; RM=[]; xyz_r=[]; inds=[];
N_new=N; arbVals=[]; norm_factor=[];


cc=zeros(Npix_im(1),Npix_im(2),nDfs);

bgVal=0;
gpuVars={'R','xyz','cp','Npix','CTF','bgVal','rowColNums','temp_image', ...
    'imref_F','mask_im','V_Fr','V_Fi','SD', ...
    'sum1','sum2','fPSD','i','dummyX','dummyY','dummyZ', ...
    'X','Y','Z','output_image','vi_rs','projPot','ew','w_det', ...
    'template','temp','rMask','RM','xyz_r','Npix_im', ...
    'cc','inds','binRange','N_new','Npix_im_norm','norm_factor'}; % 'spvFilt'

for j=1:length(gpuVars)
    eval([gpuVars{j} '=gpuArray(' gpuVars{j} ');']); wait(gdev);
end;


fprintf('estimating background...'); tic;
if( ccFlag )
    % % calculate the expected value away from one unrotated template:
    X=xyz(:,1)+cp; Y=xyz(:,2)+cp; Z=xyz(:,3)+cp;
    temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,Y,X,Z,'linear',0); wait(gdev); %b
    temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,Y,X,Z,'linear',0); wait(gdev); %b
    output_image=complex(temp_r,temp_i); wait(gdev); %b
    projPot_F=reshape(output_image,Npix,Npix); %wait(gdev);
    projPot_F(find(isnan(projPot_F)==1))=0; %wait(gdev);
    template=real(ifftn(projPot_F(idx_small_i{:})));
    bgVal=single(nanmedian(template(:)))
end;
wait(gdev);
toc

fprintf(fidLog,'now starting...\n');

% generate templates and cross-correlate them with the image:
fprintf('starting...\n'); tic
fn_mip=[searchDir fileBase fNumPadded '_mip.mrc'];

nfp=1;
highVals=[];
if( isempty(params.highThr) )
    highThr=sqrt(2).*erfcinv(nfp.*2./(prod(dims_imref).*nRotations_full.*nDfs))
end;
dummyVec=1:(Npix_im(2)*Npix_im(1));

Nsaved=10*prod(dims_imref)*nDfs; % more than enough for 1 sample/1D projection
%     Nsamples=(nRotations_full * (Npix_im(2) * Npix_im(1) * nDfs));
Nsamples=nRotations_full.* (prod(dims_imref)) .* nDfs;
arbThr_temp=sqrt(2).*erfcinv(2*Nsaved/Nsamples);
arbThr_toUse=min([arbThr_temp params.arbThr]);
%params.arbThr=arbThr_toUse;
%fprintf(fidLog,'Using arbThr of %6.3f\n',params.arbThr);

disp(datestr(now));


%% main loop:

z=gpuArray(single(randn(gather(Npix_im(1)),gather(Npix_im(2)))));
b=fourierTransformer(z);
%     Npix_im_norm=gpuArray(sqrt(b.inputSize(2).*b.halfDimSize));

tic;

CTF_full=gather(CTF);
fPSD_full=gather(fPSD);
imref_F_full=gather(imref_F);
wait(gdev);
CTF=CTF(1:cp_im(1),:,:); % untested with nonsq.
fPSD=fPSD(1:cp_im(1),:);
imref_F=imref_F(1:cp_im(1),:);
norm_factor=gpuArray(prod(Npix_im));

for i=1:nRotations
    
    qInd=fileMultiplier+i; RM=R(:,:,i)'; wait(gdev);
    xyz_r=(RM*xyz')'+cp; wait(gdev);

    temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
    temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
    output_image=complex(temp_r,temp_i); wait(gdev); %b
    
    projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
    
    template=real(ifftn(projPot_F(idx_small_i{:}))); wait(gdev);
    template=template-median(template(:)); wait(gdev);
    temp_image(rowColNums)=template(idx_small_f{:}); wait(gdev);
    
    temp=b.fwdFFT(temp_image);%./(Npix_im_norm./2); wait(gdev);
    temp=temp./(2.*sqrt(size(temp,1).*size(temp,2)));
    temp=temp.*fPSD; wait(gdev);
    for j=1:nDfs
        template_F=temp.*CTF(:,:,j); wait(gdev);
        v=sum(abs(template_F(:)).^2,1,'native'); wait(gdev); % % SD normalization
        SD(i)=sqrt(2.*v./norm_factor);
        template_F=template_F./SD(i); wait(gdev);
        
        cc_F=imref_F.*conj(template_F); wait(gdev);
        cc(:,:,j)=b.invFFT(cc_F)./sqrt(norm_factor); wait(gdev);
    end;
    
    sum1=sum1+cc; sum2=sum2+cc.^2; wait(gdev); % % moments
    newLocs=gather(find(cc>params.arbThr)); wait(gdev);
    newVals=double(gather(cc(newLocs))); wait(gdev);
    nNewVals=(length(newVals)); wait(gdev);
    newInds=repmat(qInd,nNewVals,1); wait(gdev);
    newVals_all=([newVals newLocs newInds]); wait(gdev);
    arbVals{i}=(newVals_all); wait(gdev);
    highVals=[highVals; newVals_all((newVals_all(:,1)>highThr),:)]; wait(gdev); % % highVals
    
    if( mod(i,10)==0 )
        if( mod(i,100)==0 )
            disp(max(highVals(:,1)))
            fprintf(fidLog,'\n%d/%d ',i,nRotations);
            fprintf(fidLog,'averaging %3.0f msec. per orientation',1000.*toc./100);
            fprintf('\n%d/%d ',i,nRotations);
            fprintf('averaging %3.0f msec. per orientation',1000.*toc./100);
            if( mod(i,10000)==0 )
                if( ~isempty( highVals ) )
                    [a,d,c]=ind2sub([gather(Npix_im(1)) gather(Npix_im(2)) nDfs],highVals(:,2));
                    highLocs=[a d c];
                    list_highVals=[highVals(:,3) highVals(:,1) a d c]; % index, value, row, col, page
                    fid=fopen([outputDir 'highVals_' fNumPadded '.txt'],'w');
                    for i=1:size(list_highVals,1)
                        fprintf(fid,'%i\t%8.3f\t%i\t%i\t%i\n',list_highVals(i,:));
                    end;
                    fclose(fid);
                end;
            end;
            tic;
        end;
    end;
end;
clear b

timeElapsed=toc;
fprintf(fidLog,'Done with main xcorr loop...\n');
disp(datestr(now));

for i=1:length(gpuVars)
    eval([gpuVars{i} '=gather(' gpuVars{i} ');']); wait(gdev);
end;

%
% write the output files:

if( ccFlag )
    fn_sum1=[searchDir fileBase fNumPadded '_sum.mrc'];
    fn_sum2=[searchDir fileBase fNumPadded '_ssum.mrc'];
    fn_arb=[searchDir fileBase fNumPadded '_arb.dat'];
    fn_SD=[searchDir fileBase fNumPadded '_SD.dat'];
    
    mw(single(sum1),fn_sum1,params.aPerPix);
    mw(single(sum2),fn_sum2,params.aPerPix);
    
    fid=fopen(fn_SD,'w');
    fwrite(fid,single(reshape([SD]',1,length(SD))),'single');
    fclose(fid);
    
    fprintf(fidLog,'making arbVals vector...');
    disp(datestr(now));
    fprintf(fidLog,'%s\n',datestr(now,31));
    nInVec=0;
    for i=1:nRotations
        nInVec=nInVec+size(arbVals{i},1);
    end;
    arbV=zeros(nInVec,3,'double');
    
    if( nInVec>0 )
        lastInd=0;
        for i=1:nRotations
            startInd=lastInd+1;
            nNew=size(arbVals{i},1);
            endInd=startInd+nNew-1;
            if( nNew>0 )
                arbV(startInd:endInd,:)=arbVals{i};
                lastInd=endInd;
            end;
        end;
    end;
    
    fprintf(fidLog,'done\n');
    fprintf(fidLog,'opening arbVals file...\n');
    fid=fopen(fn_arb,'w');
    fprintf(fidLog,'writing to arbVals file...\n');
    fwrite(fid,reshape(arbV',1,size(arbV,1)*3),'double');
    fprintf(fidLog,'closing arbVals file...\n');
    fclose(fid);
    fprintf(fidLog,'done with arbVals file\n');
    
else
    fn_templates=[searchDir fileBase fNumPadded '_templates.mrc'];
    mw(single(templates),fn_templates,params.aPerPix);
end;


fprintf(fidLog,'Done with job %i at %s\n',jobNum,datestr(now,31));

%% combine output and clean up:

if( fileNum==params.nCores ) % % last_job
    if( ccFlag )
        %fileTypesExpected={'mip','mipi','SD','arb','sum','ssum'};
        fileTypesExpected={'SD','arb','sum','ssum'};
    else
        fileTypesExpected={'templates'};
    end;
    
    while 1
        % % get a list of existing files to combine and do sanity check:
        nFilesExpected=params.nCores;
        fprintf(fidLog,'Looking for files from %i searches...\n',nFilesExpected);
        
        numFound=[]; fileTypesFound={};
        A=dir([searchDir 'search_*.*']);
        ctr=1;
        for i=1:length(A)
            tempNum=regexp(A(i).name,'search_(\d{4,4})_','tokens');
            if( length(tempNum)>0 )
                numFound(ctr)=str2num(char(tempNum{1}));
                tempType=regexp(A(i).name,'search_(\d{4,4})_','split');
                tempTypeParts=regexp(tempType{2},'(\.{1,1})','split');
                fileTypesFound{ctr}=tempTypeParts{1};
                ctr=ctr+1;
            end;
        end;
        
        if( ctr>(nFilesExpected.*length(fileTypesExpected)) )
            fprintf(fidLog,'combining %i files...\n',(nFilesExpected.*length(fileTypesExpected)));
            if( params.nCores>1 )
                pause(10);
            end;
            break;
        end;
        pause(1);
    end;
    
    inds={};
    for i=1:length(fileTypesExpected)
        inds{i}=find(strcmp(fileTypesFound,fileTypesExpected{i})==1);
    end;
    
    if( ccFlag )
        sumImage_group=zeros(Npix_im(1),Npix_im(2),nDfs);%size(temp,1),size(temp,2));
        ssumImage_group=zeros(Npix_im(1),Npix_im(2),nDfs);%size(temp,1),size(temp,2));
        arbVector=[];
        SDl=[];
    else
        templates_group=zeros(Npix_im(1),Npix_im(2));%size(temp,1),size(temp,2));
    end; % if( ccFlag )
    
    fid_arb=fopen([outputDir 'search_listAboveThreshold.dat'],'w');
    startStopVec=zeros(nRotations_full,2,'double');
    ctr_header=zeros(1,1,'double');
    current_byte=zeros(1,1,'double');
    av_all=[];
    
    % % read in each file-type and combine:
    for i=1:length(inds{1})
        for j=1:length(fileTypesExpected)
            fileType=fileTypesExpected{j};
            fn=[searchDir A(inds{j}(i)).name];
            switch fileType
                case 'mip'
                    mip=smap.mr(fn);
                    temp=mip>mip_group;
                    mip_inds=find(temp==1);
                    mip_group(mip_inds)=mip(mip_inds);
                case 'mipi'
                    mipi=smap.mr(fn);
                    mipi_group(mip_inds)=mipi(mip_inds);
                case 'sum'
                    sumImage=smap.mr(fn);
                    sumImage_group=sumImage_group+sumImage;
                case 'ssum'
                    ssumImage=smap.mr(fn);
                    ssumImage_group=ssumImage_group+ssumImage;
                case 'list'
                    fid=fopen(fn,'r');
                    temp=fread(fid,inf,'single');
                    fclose(fid);
                    peakVals=temp(1:2:end); peakTemp=temp(2:2:end);
                    pv=[peakVals peakTemp];
                    pvl=[pvl; pv];
                case 'SD'
                    fid=fopen(fn,'r');
                    temp=fread(fid,inf,'single');
                    fclose(fid);
                    SD=temp;
                    SDl=[SDl; SD];
                case 'arb'
                    %pause;
                    disp(i);
                    fid=fopen(fn,'r');
                    temp=fread(fid,inf,'double');
                    fclose(fid);
                    arbVals=[]; arbTemp=[]; arbInds=[];
                    if( ~isempty(temp) )
                        arbVals=temp(1:3:end); arbTemp=temp(2:3:end); arbInds=temp(3:3:end);
                        av_temp=[arbInds arbTemp arbVals];
                        fwrite(fid_arb,reshape(av_temp',1,size(av_temp,1)*3),'double');
                        av_all=[av_all; av_temp];
                        [u,uI_start]=unique(double(arbInds),'first');
                        [~,uI_stop]=unique(double(arbInds),'last');
                        startStopVec(u,1)=uI_start+ctr_header;
                        startStopVec(u,2)=uI_stop+ctr_header;
                        byte_vector=[uI_start-1].*3.*4';
                        byte_vector(:,2)=[uI_stop].*3.*4';
                        byte_vector=uint64(byte_vector);
                        byte_vector=byte_vector+current_byte;
                        startStopVec(u,:)=byte_vector;
                        current_byte=max(byte_vector(:,2));
                    end;
                    ctr_header=ctr_header+(length(arbVals));
                case 'hist'
                    fid=fopen(fn,'r');
                    N=fread(fid,inf,'int64')';
                    fclose(fid);
                    temp=reshape(N,1,nBins);
                    hist_group=hist_group+temp;
                case 'templates'
                    templates=smap.mr(fn);
                    templates_group=cat(3,templates_group,templates);
                otherwise
                    fprintf(fidLog,'file type %s not recognized...\n',fileType);
            end;
        end;
    end;
    fclose(fid_arb);
    
    if( ccFlag )
        
        nCores_opt=0;
        
        meanImage=double(sumImage_group./nRotations_full);
        squaredMeanImage=double(ssumImage_group./nRotations_full);
        SDImage=double(sqrt(squaredMeanImage-(meanImage.^2)));
        
        % try
            ai=av_all(:,1); al=av_all(:,2); av=av_all(:,3);
            
            [temp_x,temp_y,temp_z]=ind2sub([Npix_im(1) Npix_im(2) nDfs],al);
            locs=[temp_x temp_y temp_z];
            
            % % crop out any strays that arise in the padding region:
            margin_size=floor((Npix_im-dims_imref)./2)+1 + floor(cp./2)+1;
            if( margin_pix > 0 )
                margin_size=(floor(margin_pix)./2).*[1 1];
            end;
            xy_temp=locs(:,1:2);
            %                 xy_temp=randi([1 Npix_im(1)],1e5,2);
            xy_temp_abs=abs(xy_temp-repmat(cp_im,size(xy_temp,1),1));
            inds_edge=find(xy_temp_abs(:,1)>(cp_im(1)-margin_size(1)) | xy_temp_abs(:,2)>(cp_im(2)-margin_size(2)));
            inds_keep=setdiff([1:size(xy_temp_abs,1)],inds_edge);
            xy_temp=xy_temp(inds_keep,:);
            ai=ai(inds_keep); al=al(inds_keep); av=av(inds_keep);
            locs=locs(inds_keep,:);
            
            
            % %
            
            fprintf(fidLog,'pixelwise-normalizing values above threshold...\n');
            av_norm=zeros(1,length(av),'double');
            for i=1:length(av)
                av_norm(i)=(av(i)-meanImage(locs(i,1),locs(i,2),locs(i,3)))./(SDImage(locs(i,1),locs(i,2),locs(i,3)));
            end;
            
            [av_norm,sI]=sort(av_norm,'ascend');
            ai=ai(sI);
            locs=locs(sI,:);
            
            ff=zeros([Npix_im(1) Npix_im(2) nDfs],'double');
            ff_inds=ff;
            for i=1:length(sI)
                ff(locs(i,1),locs(i,2),locs(i,3))=av_norm(i);
                ff_inds(locs(i,1),locs(i,2),locs(i,3))=ai(i);
            end;
            %                 ff=smap.cropOrPad(ff,[dims_imref(1) imds_imref(2) nDfs]);
            %                 ff_inds=smap.cropOrPad(ff,[dims_imref(1) imds_imref(2) nDfs]);
            mw(single(ff),[outputDir 'search_vals.mrc'],params.aPerPix);
            mw(single(ff_inds),[outputDir 'search_qInds.mrc'],params.aPerPix);
            
            xs=gather(linspace(min(av_norm(:)),max(av_norm(:)),1e4));
            nfp=1;
            
            N_temp=hist(gather(av_norm(:)),xs);
            y_meas=sum(N_temp)-cumsum(N_temp);
            
            
            y_model=(erfc(xs./sqrt(2))./2).*Nsamples;
            SH=[xs; y_model; y_meas]';
            
            fid_SH=fopen([outputDir 'search_SH.txt'],'w');
            for jj=1:size(SH,1)
                fprintf(fid_SH,'%8.4f\t%12.3f\t%12i\n',SH(jj,:));
            end;
            fclose(fid_SH);
            
            
        % catch
        % 
        % end;
        
        try
            %                 thr=sqrt(2).*erfcinv(2./(nRotations_full * (Npix_im(1) * Npix_im(2) * nDfs)));
            thr=sqrt(2).*erfcinv(2./(nRotations_full * ((Npix_im_norm.^2) * nDfs)));
            nHighVals=length(find(av_norm>thr));
            fprintf(fidLog,'found %i values above threshold\n',nHighVals);
            if( nHighVals>0 )
                highInds=find(av_norm>thr);
                highVals=av_norm(highInds);
                highLocs=locs(highInds,:);
                highInds_q=ai(highInds);
                list_highVals=[highInds_q highVals' highLocs]; % index, value, row, col, df
                fid_highVals=fopen([outputDir 'highVals.txt'],'w');
                for jj=1:size(list_highVals,1)
                    fprintf(fid_highVals,'%i\t%8.3f\t%i\t%i\t%i\n',list_highVals(jj,:));
                end;
                fclose(fid_highVals);
            end;
        catch
            
            
        end;
        
        try
            hvf=dir([outputDir 'highVals_*.txt']);
            for jj=1:length(hvf)
                delete([outputDir hvf(jj).name]);
            end;
        catch
        end;
        
        
        % write combined files:
        fprintf(fidLog,'writing combined files...\n');
        mw(single(meanImage),[outputDir 'search_mean.mrc'],params.aPerPix);
        mw(single(SDImage),[outputDir 'search_SD.mrc'],params.aPerPix);
        fid=fopen([outputDir 'search_SDs.dat'],'w');
        fwrite(fid,single(reshape([SDl]',1,length(SDl))),'single');
        fclose(fid);
        ss=[]; qBest=[]; sI=[]; xy=[]; indVector=[];
        for i=1:nDfs
            [ss_temp,qBest_temp,sI_temp,xy_temp]=smap.clusterImByThr(ff(:,:,i),ff_inds(:,:,i),params.optThr,R_full);
            if( ~isempty(sI_temp) )
                ss=cat(2,ss,ss_temp);
                qBest=cat(3,qBest,smap.normalizeRM(qBest_temp));
                xy=cat(1,xy,xy_temp);
                indVector=[indVector; repmat(i,length(ss_temp),1)];
            end;
        end;
        
        mv=[];
        for i=1:length(ss)
            mv(i)=ss(i).MaxVal;
        end;
        
        nEl=size(xy,1)
        if( nEl>0 )
            nanmat=nan(nEl,nEl);
            
            pd_temp=squareform(pdist(xy));
            pd_mat=tril(nanmat)+pd_temp;
            pd=pd_mat(find(isnan(pd_mat(:))==0));
            qd_temp=smap.pairwiseQD(qBest,qBest);
            qd_mat=tril(nanmat)+qd_temp;
            qd=qd_mat(find(isnan(qd_mat(:))==0));
            
            
            s_mat=single(pd_mat<params.qThr & qd_mat<params.dThr);
            xy_tu=[]; q_tu=[]; hInds=[]; exclude_inds=[];
            for i=1:size(s_mat,1)
                if( nansum(s_mat(i,:))>0 )
                    ntu=[i find(s_mat(i,:)>0)];
                    mv_temp=mv(ntu);
                    hInd=ntu(find(mv_temp==max(mv_temp),1,'first'));
                    lInds=setdiff(ntu,hInd);
                    s_mat(ntu,:)=0;
                    xy_tu=[xy_tu; xy(hInd,:)];
                    q_tu=cat(3,q_tu,qBest(:,:,hInd));
                    hInds=[hInds; hInd];
                    exclude_inds=[exclude_inds lInds];
                end;
            end;
            exclude_inds=unique(exclude_inds);
            keep_inds=setdiff(1:nEl,exclude_inds);
            
            nFound=length(keep_inds);
            
            fprintf(fidLog,'starting optimization on %i particles...\n',nFound);
            fprintf('starting optimization on %i particles...\n',nFound);
            
            xy_tu=xy(keep_inds,:);
            q_tu=qBest(:,:,keep_inds);
            ss_tu=ss(keep_inds);
            indVector_tu=indVector(keep_inds);
            df_tu=params.defocus(indVector_tu,:);
            qInds_tu=[];
            for i=1:nFound
                qInds_tu(i)=ff_inds(xy_tu(i,1),xy_tu(i,2),indVector_tu(i));
            end;
            pd_tu=squareform(pdist(xy_tu));
            qd_tu=smap.pairwiseQD(q_tu,q_tu);
            nanmat=nan(length(keep_inds),length(keep_inds));
            pd_tu=tril(nanmat)+pd_tu;
            qd_tu=tril(nanmat)+qd_tu;
            
            % % parcel out jobs, write params for each:
            nCores_opt=min([nFound params.nCores])
            inds_opt=cell(1,nCores_opt); %toOpt=cell(1,nCores);
            for i=1:nCores_opt
                inds_opt{i}=[i:nCores_opt:nFound];
                if( ~isempty(inds_opt{i}) )
                    xy=xy_tu(inds_opt{i},:);
                    qInds=qInds_tu(inds_opt{i});
                    df=df_tu(inds_opt{i},:);
                    toOpt=[xy df qInds']
                    fn_opt=fullfile([scratchDir '/opt_toDo_' smap.zp(i,4) '.txt']);
                    fid_opt=fopen(fn_opt,'w');
                    fprintf(fid_opt,'%4i\t%4i\t%6.2f\t%6.2f\t%5.4f\t%i\n',toOpt');
                    fclose(fid_opt);
                end;
            end;
            if( nFound<params.nCores )
                for i=(nFound+1):params.nCores
                    fn_opt=fullfile([scratchDir '/opt_toDo_' smap.zp(i,4) '.txt']);
                    fid_opt=fopen(fn_opt,'w');
                    fclose(fid_opt);
                end;
            end;
            
        else
            fprintf(fidLog,'nothing to optimize\n');
            for i=1:params.nCores
                fn_opt=fullfile([scratchDir '/opt_toDo_' smap.zp(i,4) '.txt']);
                fid_opt=fopen(fn_opt,'w');
                fclose(fid_opt);
            end;
        end;
        
    else
        fprintf(fidLog,'writing combined templates...\n');
        templates_group=templates_group(:,:,2:end);
        mw(single(templates_group),[outputDir 'search_templates.mrc'],params.aPerPix)
    end; % if( ccFlag )
    
end; % fileNum==params.nCores

%     wait_flag=1;

%%
%     if( params.optimize_flag )

fn_opt=fullfile([scratchDir '/opt_toDo_' smap.zp(fileNum,4) '.txt']);
while 1
    if( exist(fn_opt,'file')==2 )
        fprintf(fidLog,'found file...\n');
        fprintf('found file...\n');
        pause(10);
        break;
    end;
    pause(5);
    fprintf(fidLog,'waiting...\n');
end;

try
    fid=fopen(fn_opt,'r');
    A=fscanf(fid,'%f');
    fclose(fid);
catch
    A=[];
end;

if( length(A)>=1 )
    
    
    xy_all=[A(1:6:end) A(2:6:end)]
    df_all=[A(3:6:end) A(4:6:end) A(5:6:end)]
    qInd_all=[A(6:6:end)]
    
    nParticles=length(qInd_all)
    
    fprintf(fidLog,'starting refinement on %i particles...\n',nParticles);
    fprintf('starting refinement on %i particles...\n',nParticles);
    
    
    Npix=gather(Npix);
    Npix_im=gather(Npix_im);
    Npix_im_norm=gather(Npix_im_norm);
    
    gpuVars={'R','xyz','CTF','bgVal','rowColNums','temp_image', ...
        'imref_F','mask_im','V_Fr','V_Fi','pv','pl','SD', ...
        'fPSD','i','qInd','dummyX','dummyY','dummyZ', ...
        'X','Y','Z','output_image','vi_rs','projPot','ew','w_det', ...
        'template','temp','rMask','RM','xyz_r', ...
        'cc','binRange','N_new'};
    
    for j=1:length(gpuVars)
        try
            eval([gpuVars{j} '=gpuArray(' gpuVars{j} ');']); wait(gdev);
        catch
        end;
    end;
    
    tic
    meanImage=smap.mr([outputDir 'search_mean.mrc']);
    meanImage=mean(meanImage,3);
    SDImage=smap.mr([outputDir 'search_SD.mrc']);
    SDImage=mean(SDImage,3);
    
    
    xy=[]; df=[]; qInd=[]; RM_init=[];
    for i=1:nParticles
        xy(i,:)=xy_all(i,:);
        df(i,:)=df_all(i,:);
        qInd(i)=qInd_all(i);
        RM_init(:,:,i)=R_full(:,:,qInd(i));
    end;
    fPSD_patch=gpuArray(ifftshift(imresize(fftshift(fPSD_full),Npix.*[1,1],'bilinear'))); % corner
    fPSD_patch(1,1)=0;
    
    try
        baseVec=[-params.range_degrees:params.inc_degrees:params.range_degrees];
        [xd,yd,zd]=meshgrid(baseVec,baseVec,baseVec);
        xdd=xd(:); ydd=yd(:); zdd=zd(:);
        xyz_dummy=[xdd ydd zdd];
        xyz_dummy_r=sqrt(sum(xyz_dummy.^2,2));
        itk=find(xyz_dummy_r<=params.range_degrees);
        xyz_dummy=xyz_dummy(itk,:);
        xyz_dummy_rot=xyz_dummy.*pi./180;
        R_bump=[];
        
        for i=1:size(xyz_dummy,1)
            R_bump(:,:,i)=rotationVectorToMatrix(xyz_dummy_rot(i,:));
        end;
        fprintf(fidLog,'refining over %3.2f degree range, %3.2f degree increments\n',params.range_degrees,params.inc_degrees);
        
    catch
        fprintf(fidLog,'refining with default rotation set...\n');
        R_bump=smap.normalizeRM(smap.readRotationsFile('~/smap_ij/rotation/qBump_2deg_0.5degInc.txt'));
    end;
    
    nBump=size(R_bump,3);
    Rtu=[];
    
    xyz=gpuArray(xyz);
    %         imref=imref_orig;
    
    patch_mask=single(smap.rrj(ones(Npix,Npix)).*Npix)<=5;
    inds_mask=find(patch_mask==1);
    
    fprintf(fidLog,'flag 1\n');
    cc_max_opt=zeros(nParticles,1);
    q_best_all=[];
    for i=1:nParticles
        tic
        patch=smap.cropOrPad(imref,Npix.*[1 1],xy(i,:)); % yes, PSD-filtered
        patch_F=(fftn(ifftshift(gpuArray(patch)))); % corner
        patch_mean=smap.cropOrPad(meanImage,Npix.*[1 1],xy(i,:));
        patch_SD=smap.cropOrPad(SDImage,Npix.*[1 1],xy(i,:));
        patch_mean_masked=patch_mean(inds_mask);
        patch_SD_masked=patch_SD(inds_mask);
        
        fprintf(fidLog,'flag 2\n');
        
        % % with this:
        CTF_patch=smap.ctf(df(i,:),Npix.*[1,1]);
        CTF_patch=ifftshift(imag(CTF_patch));
        temp_F=patch_F;%(:,:,i);
        patchref_F=temp_F./std(temp_F(:));
        
        RM=RM_init(:,:,i)'; %wait(gdev);
        
        xyz_r=(RM*xyz')'+cp; wait(gdev);
%         xyz_r=(gather(RM)*gather(xyz'))'+cp; wait(gdev); % 031623: no clue why this is needed for A5000 boards:

        temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        output_image=complex(temp_r,temp_i); wait(gdev); %b
        
        projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
        projPot_F=projPot_F(idx_small_i{:});
        projPot_F(1,1)=0;
        temp=projPot_F.*fPSD_patch.*CTF_patch;
        temp=temp./std(temp(:));
        template_F=conj(temp);
        cc_F=arrayfun(@times,patchref_F,template_F); wait(gdev); % % xcorr
        cc_temp=fftshift(real(ifftn(cc_F))).*Npix; wait(gdev);
        cc=(cc_temp(inds_mask)-patch_mean_masked)./patch_SD_masked; wait(gdev);
        
        
        fprintf(fidLog,'flag 4\n');
        
        fprintf(fidLog,'init: %f\n',max(cc(:)));
        fprintf('init: %f\n',max(cc(:)));
        
        toc; tic
        
        
        % % scan the defocus:
        df1=df(i,1:2)-50;
        df2=df(i,1:2)+50;
        df_test_1=[df1(1):(params.aPerPix).*2:df2(1)]'; % 051921
        df_test_2=[df1(2):(params.aPerPix).*2:df2(2)]'; % 051921
        df_test_ast=repmat(df(i,3),size(df_test_1,1),1);
        df_test=[df_test_1 df_test_2 df_test_ast];
        nDfs_patch=size(df_test,1);
        
        temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        output_image=complex(temp_r,temp_i); wait(gdev); %b
        
        projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
        projPot_F=projPot_F(idx_small_i{:});
        projPot_F(1,1)=0;
        temp_unmod=projPot_F.*fPSD_patch;
        
        CTF_patch=imag(smap.ctf(df_test,Npix.*[1,1]));
        
        template_F=gpuArray.zeros(Npix,Npix);
        for j=1:nDfs_patch
            temp=temp_unmod.*ifftshift(CTF_patch(:,:,j));
            temp(1,1)=0;
            temp=temp./std(temp(:)); wait(gdev);
            template_F(:,:,j)=conj(temp);
        end;
        
        fprintf(fidLog,'flag 5\n');
        
        cc_max=[];
        for j=1:nDfs_patch
            cc_F=arrayfun(@times,patchref_F,template_F(:,:,j));
            cc_temp=fftshift(real(ifftn(cc_F))).*Npix; wait(gdev);
            cc=(cc_temp(inds_mask)-patch_mean_masked)./patch_SD_masked; wait(gdev);
            cc_max(j)=gather(max(cc(:)));
        end;
        fprintf(fidLog,'defocus: %f\n',max(cc_max))
        fprintf('defocus: %f\n',max(cc_max))
        
        ind_best=find(cc_max==max(cc_max),1,'first')
        df_best=df_test(ind_best,:);
        CTF_patch=ifftshift(imag(smap.ctf(df_best,Npix.*[1,1])));
        
        
        toc; tic
        
        
        % % now do angles:
        for j=1:nBump
            Rtu(:,:,j)=R_bump(:,:,j)*RM_init(:,:,i);
        end;
        Rtu=smap.normalizeRM(Rtu);
        
        cc_max=[];
        for j=1:nBump
            RM=Rtu(:,:,j)';
            
            xyz_r=(RM*xyz')'+cp; wait(gdev);
%             xyz_r=(gather(RM)*gather(xyz'))'+cp; wait(gdev); % 031623: no clue why this is needed for A5000 boards:
            
            temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
            temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
            output_image=complex(temp_r,temp_i); wait(gdev); %b
            projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
            
            projPot_F=projPot_F(idx_small_i{:});
            projPot_F(1,1)=0;
            temp=projPot_F.*fPSD_patch.*CTF_patch;
            temp=temp./std(temp(:));
            template_F=conj(temp);
            
            cc_F=arrayfun(@times,patchref_F,template_F); wait(gdev); % % xcorr
            cc_temp=fftshift(real(ifftn(cc_F))).*Npix; wait(gdev);
            cc=(cc_temp(inds_mask)-patch_mean_masked)./patch_SD_masked; wait(gdev);
            
            cc_max(j)=gather(max(cc(:)));
            
        end;
        fprintf(fidLog,'angle: %f\n',max(cc_max))
        fprintf('angle: %f\n',max(cc_max))
        
        fprintf(fidLog,'flag 6\n');
        
        ind_best=find(cc_max==max(cc_max),1,'first')
        q_best=Rtu(:,:,ind_best);
        
        RM=q_best';
        xyz_r=(RM*xyz')'+cp; wait(gdev);
%         xyz_r=(gather(RM)*gather(xyz'))'+cp; wait(gdev); % 031623: no clue why this is needed for A5000 boards:

        toc; tic
        
        
        % % scan the defocus again:
        df1=df_best(1:2)-50;
        df2=df_best(1:2)+50;
        df_test_1=[df1(1):(params.aPerPix):df2(1)]'; % 051921
        df_test_2=[df1(2):(params.aPerPix):df2(2)]'; % 051921
        df_test_ast=repmat(df_best(3),size(df_test_1,1),1);
        df_test=[df_test_1 df_test_2 df_test_ast];
        nDfs_patch=size(df_test,1);
        
        temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        output_image=complex(temp_r,temp_i); wait(gdev); %b
        projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
        
        projPot_F=projPot_F(idx_small_i{:});
        projPot_F(1,1)=0;
        temp_unmod=projPot_F.*fPSD_patch;
        
        fprintf(fidLog,'flag 7\n');
        
        CTF_patch=imag(smap.ctf(df_test,Npix.*[1,1]));
        
        template_F=gpuArray.zeros(Npix,Npix);
        for j=1:nDfs_patch
            temp=temp_unmod.*ifftshift(CTF_patch(:,:,j));
            temp(1,1)=0;
            temp=temp./std(temp(:)); wait(gdev);
            template_F(:,:,j)=conj(temp);
        end;
        
        cc_max=[];
        for j=1:nDfs_patch
            cc_F=arrayfun(@times,patchref_F,template_F(:,:,j));
            cc_temp=fftshift(real(ifftn(cc_F))).*Npix; wait(gdev);
            cc=(cc_temp(inds_mask)-patch_mean_masked)./patch_SD_masked; wait(gdev);
            cc_max(j)=gather(max(cc(:)));
        end;
        fprintf(fidLog,'defocus: %f\n',max(cc_max))
        fprintf('defocus: %f\n',max(cc_max))
        
        ind_best=find(cc_max==max(cc_max),1,'first')
        df_best=df_test(ind_best,:);
        CTF_best=ifftshift(imag(smap.ctf(df_best,Npix.*[1,1])));
        toc; tic
        
        % % now get the final xcorr value using the full image:
        xyz_r=((q_best')*xyz')'+cp;
%         xyz_r=(gather(q_best')*gather(xyz'))'+cp; wait(gdev); % 031623: no clue why this is needed for A5000 boards:
        
        temp_r = interpn(dummyX,dummyY,dummyZ,V_Fr,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        temp_i = interpn(dummyX,dummyY,dummyZ,V_Fi,xyz_r(:,2),xyz_r(:,1),xyz_r(:,3),'linear',0); wait(gdev); %b
        output_image=complex(temp_r,temp_i); wait(gdev); %b
        projPot_F=reshape(output_image,Npix,Npix); wait(gdev);
        
        projPot_F=projPot_F(idx_small_i{:});
        projPot_F(1,1)=0;
        temp=projPot_F.*fPSD_patch.*CTF_best;
        temp=fftshift(real(ifftn(temp)));
        temp=smap.cropOrPad(temp,[Npix_im(1) Npix_im(2)],0);
        template_F=fftn(ifftshift(temp));
        template_F=template_F./std(template_F(:));
        
        cc_F=arrayfun(@times,imref_F_full,conj(template_F)); wait(gdev); % % xcorr
        cc_final=fftshift(real(ifftn(cc_F))).*sqrt(prod(Npix_im));
        cc=(cc_final-meanImage)./SDImage; wait(gdev);
        
        cc=smap.cropOrPad(smap.cropOrPad(cc,dims_imref),Npix_im,0);
        
        cc_best=nanmax(cc(:))
        [temp_x,temp_y]=find(cc==nanmax(cc(:)),1,'first');
        
        fprintf(fidLog,'final (full): %f\n',cc_best)
        fprintf('final (full): %f\n',cc_best)
        
        toc; tic
        
        outVals(i,:)=[gather(cc_best) gather(temp_x) gather(temp_y) gather(df_best) gather(double(quaternion.rotationmatrix(q_best))')]
        q_best_all(:,:,i)=gather(q_best);        
    end;
    
    
    try
        if( params.debugFlag )
            fprintf(fidLog,'Keeping opt files...\n');
        else
            fprintf(fidLog,'Deleting opt files...\n');
            delete(fn_opt);
        end;
    end;
    
    fn_opt_done=strrep(fn_opt,'opt_toDo_','opt_done_');
    fid_opt=fopen(fn_opt_done,'w');
    fprintf(fid_opt,'%6.3f\t%4i\t%4i\t%6.2f\t%6.2f\t%4.3f\t%7.6f\t%7.6f\t%7.6f\t%7.6f\n',outVals');
    fclose(fid_opt);
    
    if( params.aPerPix_search ~= aPerPix_orig )
        coords_bin=outVals(:,2:3);
        
        % % pixel pitches before and after binning:
        aPerPix_0=aPerPix_orig;%params.aPerPix./params.searchBin;% 1.032;
        aPerPix_b=params.aPerPix; %1.5;

        % % original dimensions before padding:
        h_0=size(imref_orig,1); %1728;
        w_0=size(imref_orig,2) %3456;
        
        % % binned dimensions before padding:
        h_b=dims_imref(1); %1188;
        w_b=dims_imref(2); %2377;
        
        dummy=smap.resizeForFFT(imref_orig,'pad',0);
        cp_0=floor(size(dummy)./2)+1
        cp_0_row=cp_0(1);
        cp_0_col=cp_0(2);
        R_b=coords_bin(:,1);
        C_b=coords_bin(:,2);
        
        cp_b=floor(Npix_im./2)+1
        cp_b_row=cp_b(1);
        cp_b_col=cp_b(2);
        xy_A_b=([R_b C_b]-repmat(cp_b,size(R_b,1),1)).*aPerPix_b;
        
        c_0=round( (C_b-cp_b_col).*(aPerPix_b./aPerPix_0)+cp_0_col );
        r_0=round( (R_b-cp_b_row).*(aPerPix_b./aPerPix_0)+cp_0_row );
        coords_orig=[r_0 c_0]
        
        params.aPerPix=aPerPix_orig;
        
    else
        coords_orig=outVals(:,2:3);
    end;
    
    fprintf(fidLog,'ready to re-load board...\n');
    
    varNames={'num_id','FOV_id','model_id','search_id','label', ...
        'peak','xy','q','df','T_min','xy_global','df_image'};
    nVars=length(varNames);
    theTable=array2table(zeros(0,nVars),'VariableNames',varNames);
    
    part_varNames={'num_id', 'FOV_id', 'model_id', 'search_id', 'label', ...
        'peak', 'xy', 'q', 'df', 'T_min', 'xy_global', ...
        'df_image'};
    temp_part=cell2table(cell(1,length(part_varNames)), 'VariableNames', part_varNames);
    
    try
        [aa,bb,cc]=fileparts(params.imageFile);
        temp=regexp(bb,'_','split');
        FOV_id=char(join(temp(1:3),'_'));
        temp=split(FOV_id,'_');
        FOV_date=char(temp(1));
        [aa,bb,cc]=fileparts(params.modelFile);
        model_id=bb;
        [aa,bb,cc]=fileparts(params.outputDir);
        temp=regexp(bb,'-','split');
        search_id=char(temp(end));
%         q_temp=smap.normalizeRM(squeeze(RotationMatrix(quaternion(outVals(:,(end-3):end)))));
    catch
        FOV_id=''; model_id=''; search_id='';
    end;
    try
        im_T=smap.mr(['~/data/' char(FOV_date) '/est/' char(FOV_id) '_T.mrc']);
    catch
        im_T=nan(Npix_im);
    end;
    
    fprintf(fidLog,'prepping table...\n');
    part=[];
    for j=1:size(outVals,1)
        temp_part.num_id=0;
        temp_part.FOV_id=FOV_id;
        temp_part.model_id=model_id;
        temp_part.search_id=search_id;
        temp_part.label='arb';
        temp_part.peak=outVals(j,1);
        temp_part.xy=coords_orig(j,:);%outVals(j,2:3);
%         temp_part.q={q_temp(:,:,j)};
        temp_part.q={q_best_all(:,:,j)};
        temp_part.df=outVals(j,4:6);
        try
            temp_part.T_min=im_T(temp_part.xy(1),temp_part.xy(2));
        catch
            temp_part.T_min=nan;
        end;
        temp_part.xy_global=[0 0]; % fix
        temp_part.df_image={[]}; % fix
        
        part=[part; temp_part];
    end;
    
    % % % % %
    % START INSERT OPTIMIZATION HERE
    % % % % %
    
    global Npix cp;
    global xd yd
    global SPV;
    global xyz V_Fr V_Fi dummyX
    global Rref
    global imref;
    global nfIm meanImage SDImage fPSD fPSD_patch inds_mask nfIm_unmod
    
    fn_out=fullfile([scratchDir '/particles_' smap.zp(jobNum,4) '.mat']);
    fn_patch_out=fullfile([scratchDir '/particles_' smap.zp(jobNum,4) '.mrc']);
    fn_template_out=fullfile([scratchDir '/templates_' smap.zp(jobNum,4) '.mrc']);
    fn_frealign_out=fullfile([scratchDir '/particles_' smap.zp(jobNum,4) '.par']);
    fn_frealign_CC_out=fullfile([scratchDir '/CCs_' smap.zp(jobNum,4) '.txt']);
    
    fprintf(fidLog,'starting true optimization...\n');
    
    SPV=smap.mr(params.modelFile);
    if( ~isempty(params.mask_params) )
        [SPV,~]=smap.mask_a_volume(SPV,params.mask_params);
        fprintf(fidLog,'masked\n');
    end;
    Npix=size(SPV,1);
    [x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
    cp=floor(Npix./2)+1;
    x0=x(:,:,cp)-cp;
    y0=y(:,:,cp)-cp;
    z0=zeros(Npix,Npix);
    
    xyz=[x0(:) y0(:) z0(:)];
    xyz_temp=[x0(:) y0(:) z0(:)];
    xyz=gpuArray(xyz_temp);
    dummyX=gpuArray([1:Npix]);
    
    clear x y z x0 y0 z0 xyz_temp;
    V_F=fftshift(fftn(ifftshift(SPV)));
    V_F(cp,cp,cp)=0;
    
    V_Fr=gpuArray(single(real(V_F)));
    V_Fi=gpuArray(single(imag(V_F)));
    clear V_F SPV
    
    mask=smap.rrj(ones(Npix,Npix)).*Npix;
    inds_mask=find(mask(:)>(Npix/2));
    
    nfIm=smap.mr(params.imageFile);
    
    meanImage=smap.mr(fullfile([outputDir '/search_mean.mrc']));
    meanImage=mean(meanImage,3);
    SDImage=smap.mr(fullfile([outputDir '/search_SD.mrc']));
    SDImage=mean(SDImage,3);
    
    Npix_orig=size(nfIm);
    nfIm=smap.resizeForFFT(nfIm,'pad',0);
    nfIm_unmod=nfIm;
    nfIm=nfIm-mean(nfIm(:));
    [fPSD,nfIm]=smap.psdFilter(nfIm,'sqrt');
    %         fPSD_patch=smap.resize_F(fPSD,Npix/fullX,'newSize');
    fPSD_patch=gpuArray(imresize(fPSD,Npix.*[1,1],'bilinear')); % corner
    
    if( params.maskCrossFlag )
        nfIm=smap.mask_central_cross(nfIm);
    end;
    
    fPSD_patch(cp,cp)=0;
    Npix_im=size(nfIm);
    cp_im=floor(Npix_im./2)+1;
    
    nfIm=smap.cropOrPad(nfIm,Npix_orig);
    nfIm=smap.nm(nfIm);
    nfIm=smap.cropOrPad(nfIm,Npix_im,0);
    
    meanImage=smap.cropOrPad(meanImage,dims_imref);
    %         meanImage=smap.resize_F(meanImage,params.searchBin,'newSize');
    meanImage=imresize(meanImage,Npix_orig,'bilinear'); % corner
    %         meanImage=smap.cropOrPad(meanImage,size(imref_orig));
    meanImage=smap.cropOrPad(meanImage,size(nfIm),0);
    
    SDImage=smap.cropOrPad(SDImage,dims_imref);
    %         SDImage=smap.resize_F(SDImage,params.searchBin,'newSize');
    SDImage=imresize(SDImage,Npix_orig,'bilinear'); % corner
    %         SDImage=smap.cropOrPad(SDImage,size(imref_orig),mean(SDImage(:)));
    SDImage=smap.cropOrPad(SDImage,size(nfIm),1);
    
    fprintf(fidLog,'reading search moments...\n');
    
    tableref_new=[]; os_all=[];
    patch_stack=[]; template_stack=[];
    ctr=1;
    for j=1:size(part,1)
        fprintf(fidLog,'particle %i...',j);
        tic
        tableref=part(j,:);
        [temp,output_struct,patchref,templateref]=optimize_xcorr(tableref);
        disp(temp);
        tableref_new=[tableref_new; temp];
        disp([num2str(temp.peak_opt-temp.peak) ' ' num2str(ctr)])
        patch_stack=cat(3,patch_stack,patchref);
        template_stack=cat(3,template_stack,templateref);
        fprintf(fidLog,'done\n');
        ctr=ctr+1;
        toc
    end;
    
    fprintf(fidLog,'saving .mat file...\n');
    save(fn_out,'tableref_new');
    fprintf(fidLog,'writing particle stacks...\n');
    smap.mw(single(patch_stack),fn_patch_out,params.aPerPix);
    smap.mw(single(template_stack),fn_template_out,params.aPerPix);
    
    % % % % %
    % END INSERT OPTIMIZATION HERE
    % % % % %
    
    q_offset=[1 0 0; 0 -1 0; 0 0 -1];
    fid_rec=fopen(fn_frealign_out,'w');
    fid_CC=fopen(fn_frealign_CC_out,'w');
    
    for ctr=1:size(part,1)
        part_new=tableref_new(ctr,:);
        theQ=part_new.q_opt{1};
        theDf=part_new.df;
        
        dfs_file=[theDf(1:2).*10 theDf(3).*(-180./pi)];
        theQ=smap.normalizeRM(theQ*q_offset);
        q_file=smap.smap2frealign(theQ);
        
        newLine=[smap.zp(ctr,7,' ') ...
            sprintf('%8.2f%8.2f%8.2f',q_file) '  ' ...
            smap.zp(sprintf('%3.2f',0.0),8,' ') '  ' ... % shx
            smap.zp(sprintf('%3.2f',0.0),8,' ') '  ' ... % shy
            smap.zp(sprintf('%5.0f',5000./(params.aPerPix./10)),6,' ') '  ' ... % mag
            smap.zp(sprintf('%4.0f',1),4,' ') '  ' ... % film
            smap.zp(sprintf('%3.1f',dfs_file(1)),7,' ') '  ' ...
            smap.zp(sprintf('%3.1f',dfs_file(2)),7,' ') '  ' ...
            smap.zp(sprintf('%3.2f',dfs_file(3)),6,' ') '  ' ...
            smap.zp(sprintf('%3.2f',100.00),6,' ') '  ' ... % occ
            smap.zp(sprintf('%3.0f',0),8,' ') '  ' ... % logP
            smap.zp(sprintf('%3.2f',0.5),9,' ') '  ' ... % sigma
            smap.zp(sprintf('%3.2f',0.0),6,' ') '  ' ... % score
            smap.zp(sprintf('%3.2f',0.0),6,' ')]; % change
        fprintf(fid_rec,'%s\n',newLine);
        newLine=[smap.zp(ctr,7,' ') sprintf('  %6.3f',part_new.peak_opt)];
        fprintf(fid_CC,'%s\n',newLine);
    end;
    fclose(fid_rec);
    fclose(fid_CC);
else
    
    try
        fprintf(fidLog,'nothing to optimize (job %i)\n',fileNum);
    catch
        
    end;
    fprintf('nothing to optimize (job %i)\n',fileNum);
    
end;
%     end;

if( fileNum==params.nCores )
    
    params.aPerPix=aPerPix_orig;
    
    if( nCores_opt>0 )
        
        while 1
            % % get a list of existing files to combine and do sanity check:
            nFilesExpected=nCores_opt;
            fprintf(fidLog,'Looking for files from %i optimizations...\n',nFilesExpected);
            
            
            numFound=[]; fileTypesFound={};
            A=dir([scratchDir '/opt_done_*.*']);
            B=dir([scratchDir '/particles_*.mat']);
            C=dir([scratchDir '/particles_*.mrc']);
            D=dir([scratchDir '/templates_*.mrc']);
            E=dir([scratchDir '/particles_*.par']);
            F=dir([scratchDir '/CCs_*.txt']);
            if( length(A)>=nFilesExpected & length(B)>=nFilesExpected & length(C)>=nFilesExpected & length(D)>=nFilesExpected ...
                    & length(E)>=nFilesExpected & length(F)>=nFilesExpected )
                fprintf(fidLog,'combining %i files...\n',nFilesExpected);
                
                % % 042219: 5 => 10
                pause(10);
                
                
                break;
            end;
            pause(1);
        end;
        
        z=[];
        for i=1:length(A)
            fid_opt_done=fopen([scratchDir '/' A(i).name],'r');
            temp=fscanf(fid_opt_done,'%f');
            if( ~isempty(temp) )
                z=[z; temp];
            end;
            fclose(fid_opt_done);
        end;
        
        inc=10;
        vals_final=z(1:inc:end);
        xy_final=[z(2:inc:end) z(3:inc:end)];
        df_final=[z(4:inc:end) z(5:inc:end) z(6:inc:end)];
        q_final=[z(7:inc:end) z(8:inc:end) z(9:inc:end) z(10:inc:end)];
        
        out_final=[vals_final xy_final df_final q_final];
        
        fn_final=[outputDir 'list_final.txt'];
        fid_final=fopen(fn_final,'w');
        fprintf(fid_final,'%6.3f\t%i\t%i\t%6.2f\t%6.2f\t%4.3f\t%7.6f\t%7.6f\t%7.6f\t%7.6f\n',out_final');
        fclose(fid_final);
        
        stack=[];
        for i=1:length(C)
            fn=[scratchDir '/' C(i).name]
            fprintf(fidLog,'opening %s...\n',fn);
            temp=smap.mr(fn);
            stack=cat(3,stack,temp);
        end;
        smap.mw(single(stack),[outputDir 'particles.mrc'],params.aPerPix);
        
        stack=[];
        for i=1:length(D)
            fn=[scratchDir '/' D(i).name]
            fprintf(fidLog,'opening %s...\n',fn);
            temp=smap.mr(fn);
            stack=cat(3,stack,temp);
        end;
        smap.mw(single(stack),[outputDir 'templates.mrc'],params.aPerPix);
        
        z=[]; part=[];
        for i=1:length(B)
            fn=[scratchDir '/' B(i).name]
            fprintf(fidLog,'opening %s...\n',fn);
            load(fn,'tableref_new');
            part=[part; tableref_new];
        end;
        part.num_id=[1:size(part,1)]';
        save([outputDir 'particles.mat'],'part');
        
        fn_frealign=fullfile([outputDir '/particles.par']);
        for i=1:length(E)
            fn=[scratchDir '/' E(i).name];
            words=['cat ' fn ' >> ' fn_frealign ];
            [~,resp]=system(words);
        end;
        
        
        % % reformat for frealign (sequential particle #s...there is probably a more elegant way to do this in awk):
        list_here = [];
        fid_frealign=fopen(fn_frealign,'r');
        lineCtr=1;
        while 1
            line = fgetl(fid_frealign);
            if ~ischar(line), break, end
            if( (isempty(strfind(line,'#')) ) & ~isempty(line) )
                list_here{lineCtr}=line;
                lineCtr=lineCtr+1;
            end;
        end;
        fclose(fid_frealign);
        outmat=str2num((cell2mat(list_here')));
        
        fid_frealign=fopen(fn_frealign,'w');
        fprintf(fid_frealign,'C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LogP      SIGMA   SCORE  CHANGE\n');
        for ctr=1:size(outmat,1)
            newLine=[smap.zp(ctr,7,' ') ...
                sprintf('%8.2f%8.2f%8.2f',outmat(ctr,2:4)) '  ' ...
                smap.zp(sprintf('%3.2f',0.0),8,' ') '  ' ... % shx
                smap.zp(sprintf('%3.2f',0.0),8,' ') '  ' ... % shy
                smap.zp(sprintf('%5.0f',outmat(ctr,7)),6,' ') '  ' ... % mag
                smap.zp(sprintf('%4.0f',1),4,' ') '  ' ... % film
                smap.zp(sprintf('%3.1f',outmat(ctr,9)),7,' ') '  ' ...
                smap.zp(sprintf('%3.1f',outmat(ctr,10)),7,' ') '  ' ...
                smap.zp(sprintf('%3.2f',outmat(ctr,11)),6,' ') '  ' ...
                smap.zp(sprintf('%3.2f',100.00),6,' ') '  ' ... % occ
                smap.zp(sprintf('%3.0f',0),8,' ') '  ' ... % logP
                smap.zp(sprintf('%3.2f',0.5),9,' ') '  ' ... % sigma
                smap.zp(sprintf('%3.2f',0.0),6,' ') '  ' ... % score
                smap.zp(sprintf('%3.2f',0.0),6,' ')]; % change
            fprintf('%s\n',newLine);
            fprintf(fid_frealign,'%s\n',newLine);
        end;
        fclose(fid_frealign);
        
        fn_CC=fullfile([outputDir '/particles.txt']);
        fid_CC=fopen(fn_CC,'w');
        fprintf(fid_CC,'#\tCC\n');
        fclose(fid_CC);
        for i=1:length(F)
            fn=[scratchDir '/' F(i).name];
            words=['cat ' fn ' >> ' fn_CC ];
            [~,resp]=system(words);
        end;
        
        list_here = [];
        fid_CC=fopen(fn_CC,'r');
        lineCtr=1;
        while 1
            line = fgetl(fid_CC);
            if ~ischar(line), break, end
            if( (isempty(strfind(line,'#')) ) & ~isempty(line) )
                list_here{lineCtr}=line;
                lineCtr=lineCtr+1;
            end;
        end;
        fclose(fid_CC);
        outmat=str2num((cell2mat(list_here')));
        
        newLine=[];
        fid_CC=fopen(fn_CC,'w');
        fprintf(fid_CC,'#\tCC\n');
        for ctr=1:size(outmat,1)
            newLine=[smap.zp(ctr,7,' ') sprintf('  %6.3f',outmat(ctr,2))];
            fprintf(fid_CC,'%s\n',newLine);
        end;
        fclose(fid_CC);
        
    end;
    
    fprintf(fidLog,'Done combining search output at %s\n',datestr(now,31));
    fprintf(fidLog,'Consolidating log files...\n');
    
    fclose(fidLog);
    fnFinal=[outputDir 'search.log'];
    fidFinal=fopen(fnFinal,'w');
    
    tline={''};
    for i=1:jobNum
        fn=[searchDir 'output_' smap.zp(i,4) '.txt'];
        try
            fid=fopen(fn,'r');
            while 1
                temp=fgets(fid);
                disp(temp);
                if ~ischar(temp)
                    break;
                else
                    fprintf(fidFinal,temp);
                end;
            end
            fclose(fid);
            delete(fn);
        catch
            tline{end+1}=['could not open ' fn];
            if( exist('fid','var') )
                if( fid>0 )
                    fclose(fid);
                end;
            end;
        end;
    end;
    
    % delete temp files:
    try
        if( params.debugFlag )
            fprintf(fidLog,'Keeping scratch files...\n');
        else
            fprintf(fidLog,'Deleting scratch files...\n');
            delete([searchDir '*.*']);
        end;
    end;
    
    [a,b,c]=fileparts(params.outputDir);
    fn_complete='queue_complete.txt';
    try
        fid=fopen(fn_complete,'a');
        fprintf(fid,'%s\n',b);
        fclose(fid);
        fprintf(fidFinal,'%s logged\n',b);
    catch
        fprintf(fidFinal,'problem logging %s\n',b);
    end;
    
    try
        if( exist(fullfile([queueDir '/done/']),'dir')<7 )
            mkdir(fullfile([queueDir '/done/']));
        end;
        movefile(paramsFile,fullfile([queueDir '/done/']));
    end
    
    fprintf(fidFinal,'*****\nFinished at %s\n*****',datestr(now,31));
    fclose(fidFinal);
    
end;

current_mode='queue'
clearvars -except jobNum current_mode gtu queueDir fn_queue gdev compile_date
clearvars -global -except gdev

return


%%
function [tableref_out,output_struct,patchref,templateref]=optimize_xcorr(tableref);

global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global Rref
global imref
global nfIm meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new
global const inds_mask bgVal params nfIm_unmod gdev

const=gpuArray(1i.*2.*pi);

% % make grid coordinates to use while applying phase shifts. Patch-sized
% % coordinates are used in the tight loop that does optimization:
[xd,yd]=meshgrid(((0:Npix-1)-Npix/2)/Npix,((0:Npix-1)-Npix/2)/Npix);
xd=gpuArray(xd); yd=gpuArray(yd);

baseDir=['~/smap_ij/dataset/'];
df_here=tableref.df;
xy_here=tableref.xy;
Rref=smap.normalizeRM(tableref.q{1}');

CTF=ifftshift(smap.ctf(tableref.df,Npix.*[1,1]));

imref=smap.nm(smap.cropOrPad(nfIm,Npix.*[1,1],xy_here));
imref_F=smap.ftj(imref);
meanref=smap.cropOrPad(meanImage,Npix.*[1,1],xy_here);
SDref=smap.cropOrPad(SDImage,Npix.*[1,1],xy_here);

% % start by centering the peak:
xyz_r=(Rref*xyz')'; wait(gdev);
% xyz_r=(gather(Rref)*gather(xyz'))'; wait(gdev); % 031623: no clue why this is needed for A5000 boards:
X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp; wait(gdev);
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0); wait(gdev); %b
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0); wait(gdev); %b
output_image=complex(temp_r,temp_i); wait(gdev); %b

projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
bgVal=nanmedian(template(inds_mask));
template=template-bgVal;
template_F=smap.ftj(template).*fPSD_patch;
template_F=template_F./std(template_F(:));
cc=smap.iftj(imref_F.*conj(template_F));
cc_corr=(cc-meanref)./SDref;
cc_F=gather(smap.ftj(cc_corr));

options = optimset('Display','off','TolFun',1e-3);

ff=@(x)-optimize_phase_shifts(cc_F,x);
[shifts,fval]=fminsearch(ff,[0 0],options);
tableref.xy_opt=(tableref.xy-shifts);
imref_s=smap.applyPhaseShifts(imref,fliplr(shifts));
meanImage_s=smap.applyPhaseShifts(meanImage,fliplr(shifts));
SDImage_s=smap.applyPhaseShifts(SDImage,fliplr(shifts));
meanref=smap.cropOrPad(meanImage_s,Npix.*[1,1],xy_here);
SDref=smap.cropOrPad(SDImage_s,Npix.*[1,1],xy_here);
imref_F=smap.ftj(imref_s);%.*CTF;

x0=[1 0 0 0 0 0];
f=@(x)-optimize_angle(imref_F,x);
[angles,fval,exitflag,output_struct]=fminsearch(f,x0,options);
RM=smap.q2R(angles(1:4))*Rref;
shifts_new=angles(5:6).*100;
tableref.q_opt{1}=RM';
shifts=shifts+shifts_new;
tableref.xy_opt=(tableref.xy-shifts);

imref=smap.nm(smap.cropOrPad(nfIm,Npix.*[1,1],xy_here));
meanref=smap.cropOrPad(meanImage,Npix.*[1,1],xy_here);
SDref=smap.cropOrPad(SDImage,Npix.*[1,1],xy_here);
imref=smap.applyPhaseShifts(imref,fliplr(shifts));
meanref=smap.applyPhaseShifts(meanref,fliplr(shifts));
SDref=smap.applyPhaseShifts(SDref,fliplr(shifts));
imref_F=smap.ftj(imref);

xyz_r=(RM*xyz')'; % 012623
% xyz_r=(gather(RM)*gather(xyz'))'; wait(gdev); % 031623: no clue why this is needed for A5000 boards:

X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0); wait(gdev); %b
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0); wait(gdev); %b
output_image=complex(temp_r,temp_i); wait(gdev); %b
projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
template=template-bgVal;

nfIm_s=smap.applyPhaseShifts(nfIm,fliplr(shifts));
nfIm_unmod_s=smap.applyPhaseShifts(nfIm_unmod,fliplr(shifts));
patchref=smap.cropOrPad(nfIm_unmod_s,Npix.*[1,1],xy_here);
templateref=gather(template);

meanImage_s=smap.applyPhaseShifts(meanImage,fliplr(shifts));
SDImage_s=smap.applyPhaseShifts(SDImage,fliplr(shifts));

template=smap.cropOrPad(template,size(nfIm,1).*[1,1],0);
template_F=smap.ftj(template).*fPSD;
template_F=template_F./std(template_F(:));

nfIm_F=smap.ftj(nfIm_s);

cc=smap.iftj(nfIm_F.*conj(template_F));

% %
cc_corr=(cc-meanImage_s)./SDImage_s;
disp(std(cc_corr(:)))

tableref.peak_opt=gather(cc_corr(xy_here(1),xy_here(2)));
tableref_out=tableref;

%%
function outref=optimize_angle(imref_F,angles);

global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global Rref
global imref
global nfIm meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new
global const inds_mask bgVal params nfIm_unmod gdev

shifts=angles(5:6).*100;

%RM=RotationMatrix(quaternion.eulerangles('xyz',angles(1:3).*pi./180))*Rref;
RM=smap.q2R(angles(1:4))*Rref;


xyz_r=(Rref*xyz')'; wait(gdev);
% xyz_r=(gather(Rref)*gather(xyz'))'; wait(gdev); % 031623: no clue why this is needed for A5000 boards:

X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp; wait(gdev);
% output_image=complex(interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0), ...
%     interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0));
% output_image = interp3gpu(dummyX,dummyX,dummyX,V_Fr,V_Fi,Y,X,Z);
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0); wait(gdev); %b
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0); wait(gdev); %b
output_image=complex(temp_r,temp_i); wait(gdev); %b
projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
template=template-bgVal;%median(template(:));
template_F=smap.ftj(template).*fPSD_patch;
template_F=template_F./std(template_F(:));

temp=smap.iftj(imref_F.*conj(template_F));
temp=(temp-meanref)./SDref;
cc_F=(smap.ftj(temp));

dphs=xd'.*(-shifts(1))+yd'.*(-shifts(2));
dphs=exp(const.*dphs);
d_done=cc_F.*dphs;
d_done=real(ifftn(ifftshift(d_done)));
outref=double(gather(d_done(1)).*Npix);


%%
function y=optimize_phase_shifts(patchref,shifts)
% % mod 060819

global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global Rref
global imref
global nfIm meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new
global const inds_mask bgVal params nfIm_unmod gdev

% global xd yd const Npix;

dphs=xd'.*(-shifts(1))+yd'.*(-shifts(2));
dphs=exp(const.*dphs);
d_done=patchref.*dphs;
d_done=real(ifftn(ifftshift(d_done)));
y=double(gather(d_done(1).*Npix));%.*Npix;

%%
function fn_SP=smappoi_calculate_SP(params,jobNum);

global gdev params

[~,fn,ext]=fileparts(params.structureFile);
fn_len=min([length(fn) 4]);
scratchDir=fullfile(params.outputDir,['scratch_' fn(1:fn_len)]);
if( exist(params.outputDir,'dir')~=7 )
    fprintf('Making new project directory... [%s]\n',params.outputDir);
    mkdir(params.outputDir);
end;
if( exist(scratchDir,'dir')~=7 )
    fprintf('Making new scratch directory... [%s]\n',scratchDir);
    mkdir(scratchDir);
end;

disp(datestr(now));
fnLog=[scratchDir '/output_' smap.zp(jobNum,4) '.txt'];
fprintf('Making new logfile... [%s]\n',fnLog);
fidLog=fopen(fnLog,'w');
fprintf(fidLog,'%s\n',datestr(now,31));

fprintf(fidLog,'job %i of %i...\n',jobNum,params.nCores);
disp(params);

aPerPix=params.aPerPix;
baseDir=[pwd '/'];
disp(datestr(now));
fileBase='EP_';
fNumPadded=smap.zp(num2str(jobNum),4);

searchDir=[scratchDir '/'];
outputDir=[params.outputDir '/'];

inc=1;
cc=smap.def_consts();
baseDir=[pwd '/'];
fn_pdb=params.structureFile;

[~,fn,ext]=fileparts(fn_pdb);
switch ext
    case {'.pdb','.PDB','.pdb1'}
        smap.read_pdb_file(fn_pdb);
    case {'.cif'}
        params_cif.CIFFile=fn_pdb;
        if( ~isempty(params.chains) )
            params_cif.chains=regexp(params.chains,' ','split');
        else
            params_cif.chains=[];
        end
        smap.read_cif_file(params_cif);
    otherwise
        fprintf('File format for %s not recognized\n',char(fn_pdb));
        fprintf(fidLog,'File format for %s not recognized\n',char(fn_pdb));
        fclose(fidLog);
        return
end;

fn_pdb=[fn ext];

if( ~isempty(params.bArb) )
    if( params.bArb == -1 )
        params.bArb=[];
    else
        fprintf(fidLog,'using imposed b-factor %5.3f for all atoms...\n',params.bArb);
        bFactor=ones(1,length(bFactor)).*params.bArb;
    end;
end;

padVal_A=5.0;

% move COM to (0,0,0):
xyz_orig=xyz;
xyz=xyz_orig-repmat((nanmean(xyz_orig,2)),1,size(xyz_orig,2));

if( isfield(params,'MW_target') )
    if( ~isempty(params.MW_target) )
        MW_target=params.MW_target
        atom_rad=sqrt(sum(xyz.^2,1));
        [atom_rad_s,sI]=sort(atom_rad,'ascend');
        dummy=[1:size(atom_rad,2)].*13.824./1000;
        ind=find(dummy<MW_target,1,'last');
        inds=sI(1:ind);
        atom_rad_s(ind)
        atomList=atomList(inds);
        bFactor=bFactor(inds);
        chainIDs=chainIDs(inds);
        atomNums=atomNums(inds);
        xyz=xyz(:,inds);
        % %
    end;
end;
xyz=xyz-repmat((nanmean(xyz,2)),1,size(xyz,2));


mm=[min(xyz,[],2) max(xyz,[],2)];
edgeSize_A=(2.*max(abs(mm(:))))+padVal_A;
edgeSize=ceil(edgeSize_A./aPerPix);
edgeSize=max([edgeSize 211]);

temp=single(smap.rrj(zeros(edgeSize,edgeSize,edgeSize)));
D=temp.*(size(temp,1)-1); % real-space values here
cp=floor(size(D,1)./2)+1;

tempVec=D(cp,:,cp);
tempVec(1:cp)=-tempVec(1:cp);
[X0,Y0,Z0]=meshgrid(tempVec,tempVec,tempVec);
[X0,Y0,Z0]=deal(X0.*aPerPix,Y0.*aPerPix,Z0.*aPerPix);

X=X0;
Y=Y0;
Z=Z0;
D=sqrt(X.^2+Y.^2+Z.^2);

nAtoms=length(atomList);
itu=jobNum:params.nCores:nAtoms;
al=char;
atu=atomList(itu);
btu=bFactor(itu);
xyz=xyz(:,itu);
for i=1:length(atu)
    temp=char(atu{i});
    al(i)=temp(1);
end;
u=unique(al);
clear uI; clear V_s;

fprintf(fidLog,'calculating potential for %d atoms...\n',length(itu));

V=zeros(size(X0));

temp=unique(sort(abs(X0(:)),'ascend'));
dx=temp(2);
atomRad=5;

m_e=cc.m_e;
h=cc.h;
q_e=cc.q_e;
c_v=cc.c_v;
IC=cc.IC;

    %         if( ~isempty(params.units) )
    %             temp=mod(params.units-1,8)+1;
    %         else
    %             temp=mod([0:(params.nCores-1)],8)+1
    %         end;
    %
    %         gtu=temp(jobNum);
    %         fprintf(fidLog,'getting gpu # %s...',num2str(gtu));
    %         tic;
    %         try
    %             gdev=gpuDevice(gtu);
    %             fprintf(fidLog,'%f seconds\n',toc);
    %             %         reset(gdev)
    %         catch
    %             fprintf(fidLog,'Failed to get gpu # %s\n',num2str(gtu));
    %             fidFail=fopen([scratchDir '/fail_' fNumPadded '.txt'],'w');
    %             fprintf(fidFail,'%s\n',datestr(now,31));
    %             exit;
    %         end;
    
    % list vars for xfer:
    gpuVars={'X','Y','Z','X0','Y0','Z0','D','V','inds','atomRad', ...
        'a','b_init','b','ctr','m_e','h','q_e','c_v','IC'}
    
    for j=1:length(gpuVars)
        if( exist(gpuVars{j},'var')==0 )
            eval([gpuVars{j} '=[];']);
        end;
        eval([gpuVars{j} '=gpuArray(' gpuVars{j} ');']); wait(gdev);
    end;

update_interval=200;

for ctr=1:length(itu)
    X=X0-xyz(1,ctr);
    Y=Y0-xyz(2,ctr);
    Z=Z0-xyz(3,ctr);
    D=sqrt(X.^2+Y.^2+Z.^2);
    inds=find(D<=atomRad); % indices for which to calculate potential
    
    el=al(ctr);
    [a,b_init]=smap.parameterizeSF(el);
    b=b_init+btu(ctr)+16.*(aPerPix.^2);
    lead_term=((16.*pi.^(5/2).*(h./(2.*pi)).^2)./(m_e.*q_e)).*1e20; % does not use relativistic electron mass
    sum_term=gpuArray.zeros(size(inds,1),5);
    for i=1:5
        sum_term(:,i)=(a(i)./(b(i).^(3./2))).*exp(-((4.*pi.^2).*(D(inds).^2))./b(i));
    end;
    sum_term=sum(sum_term,2);
    V(inds)=V(inds)+lead_term.*sum_term;
    
    if( mod(ctr,update_interval)==0 )
        fprintf('%d/%d\n',ctr,length(itu));
        fprintf(fidLog,'%d/%d\n',ctr,length(itu));
    end;
end;

for i=1:length(gpuVars)
    eval([gpuVars{i} '=gather(' gpuVars{i} ');']); wait(gdev);
end;

if( ~isempty( params.modelName ) )
    fn=strtrim(char(params.modelName));
else
    [~,fn]=fileparts(params.structureFile);
end;

fn_out=[searchDir fn '_EP_' fNumPadded '.mrc'];
fn_EP=[outputDir fn '_EP.mrc'];
fn_SP=strrep(fn_EP,'_EP.mrc','_SP.mrc');

smap.mw(single(V),fn_out,aPerPix);

fprintf(['done with potential calculation at ' datestr(now,31) '\n']);
interval=0.01;
nextFile=1;


if( jobNum==params.nCores )
    nFilesExpected=params.nCores;%*2;
    fn_pdb=params.structureFile;
    
    while 1
        fprintf(fidLog,'Looking for %i %s_EP files...\n',nFilesExpected,fn);
        
        fnList={};
        numFound=[]; fileTypesFound={};
        A=dir([searchDir fn '_EP_*.mrc']);
        ctr=1;
        for i=1:length(A)
            tempNum=regexp(A(i).name,[fn '_EP_(\d{4,4})'],'tokens');
            if( length(tempNum)>0 )
                numFound(ctr)=str2num(char(tempNum{1}));
                fnList{ctr}=[searchDir A(i).name];
                fprintf(fidLog,'%s\n',fnList{ctr});
                ctr=ctr+1;
            end;
        end;
        if( ctr>nFilesExpected )
            break;
        end;
        pause(1);
        
    end;
    
    fprintf(fidLog,'reading files...\n');
    for i=1:length(fnList)
        try
            if( i==1 )
                temp=smap.mr(fnList{i});
                EPV=zeros(size(temp,1),size(temp,2),size(temp,3));
            end;
            temp=smap.mr(fnList{i});
            EPV=EPV+temp;
        catch
            fprintf(fidLog,'problem reading file %i\n',i)
        end;
    end;
    
    %pause;
    
    wPot=4.871; % Rullgard's calculation
    out=EPV+ones(size(EPV)).*wPot;
    EPV=out;
    
    fprintf(fidLog,'writing combined electrostatic potential...\n');
    smap.mw(single(EPV),fn_EP,params.aPerPix);
    fprintf(fidLog,'done combining electrostatic potentials at %s\n',datestr(now,31));
    
    cc=smap.def_consts();
    
    SPV=EPV;
    try
        pd=smap.particleDiameter(SPV);%,0.05)
        edgeSize=size(smap.resizeForFFT(zeros(round(2.5*pd),round(2.5*pd)),'crop'),1);
    catch
        disp('problem estimating particle diameter...');
        pd=size(SPV,1)
        edgeSize=2.5*pd;
    end;
    if( edgeSize>params.edge_max )
        edgeSize=params.edge_max;
        fprintf(fidLog,'using edge_max for padding (%i^3)...\n',params.edge_max);
    end;
    edgeSize=max([edgeSize max(size(SPV))]);
    fprintf(fidLog,'padding to %i^3 volume...\n',edgeSize);
    SPV_out=single(smap.cropOrPad(SPV,[edgeSize,edgeSize,edgeSize],wPot));
    
    fprintf(fidLog,'computing phase shifts...\n');
    dx=params.aPerPix./1e10;
    
    lambda = h/sqrt(q_e*params.V_acc*m_e*(q_e/m_e*params.V_acc/c_v^2 + 2 ));
    k=2.*pi./lambda;
    SPV_out_2=SPV_out.*IC.*dx./(2.*k);
    
    %         %%
    fprintf(fidLog,'centering scattering potential...\n');
    
    inref=SPV_out_2;
    
    bg_val=mode(inref(:));
    inref=inref-bg_val;
    edge_size=size(inref,1);
    cp=floor(edge_size./2)+1;
    dummy=[-(cp-1):(cp-(2-mod(edge_size,2)))];
    [X,Y,Z]=meshgrid(dummy,dummy,dummy);
    
    M=sum(inref(:));
    COM_x=dot(X(:),inref(:))/M;
    COM_y=dot(Y(:),inref(:))/M;
    COM_z=dot(Z(:),inref(:))/M;
    disp([COM_x COM_y COM_z]);
    
    outref=smap.applyPhaseShifts(inref,-[COM_y COM_x COM_z]);
    
    M=sum(outref(:));
    COM_x=dot(X(:),outref(:))/M;
    COM_y=dot(Y(:),outref(:))/M;
    COM_z=dot(Z(:),outref(:))/M;
    
    disp([COM_x COM_y COM_z])
    
    outref=outref+bg_val;
    
    SPV_out_2=outref;
    
    %         %%
    
    fprintf(fidLog,'writing scattering potential...\n');
    
    smap.mw(single(real(SPV_out_2)),fn_SP,params.aPerPix);
    
    fprintf(fidLog,'deleting scratch files...\n');
    for i=1:length(fnList)
        delete(fnList{i});
    end;
    
    
    fprintf(fidLog,'Done combining intermediate output at %s\n',datestr(now,31));
    fprintf(fidLog,'Consolidating log files...\n');
    
    fclose(fidLog);
    
    fnFinal=strrep(fn_SP,'_SP.mrc','.log');
    fidFinal=fopen(fnFinal,'w');
    
    tline={''};
    for i=1:jobNum
        fn=[searchDir 'output_' smap.zp(i,4) '.txt'];
        try
            fid=fopen(fn,'r');
            while 1
                temp=fgets(fid);
                disp(temp);
                if ~ischar(temp)
                    break;
                else
                    fprintf(fidFinal,temp);
                end;
            end
            fclose(fid);
            delete(fn);
        catch
            tline{end+1}=['could not open ' fn];
            if( exist('fid','var') )
                if( fid>0 )
                    fclose(fid);
                end;
            end;
        end;
    end;
    
    
    fprintf(fidFinal,'*****\nFinished at %s\n*****',datestr(now,31));
    fclose(fidFinal);
    
    try
        rmdir(scratchDir);
        if( exist(fullfile([queueDir '/done/']),'dir')<7 )
            mkdir(fullfile([queueDir '/done/']));
        end;
        movefile(paramsFile,fullfile([queueDir '/done/']));
        
    catch
        
    end
    
end;

