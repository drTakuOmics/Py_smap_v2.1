function smappoi_search_local(paramsFile,this_job_num,varargin);
% % function smappoi_search_local(paramsFile,num_jobs,this_job_num,varargin);
% %%
% compile with:
% /misc/local/matlab-2019a/bin/mcc -mv -I ~/smappoi_min/v2.1/src/smap_tools/ -I ~/smappoi_min/v2.1/src/emClarity_FFT/ -R -singleCompThread smappoi_search_local.m
%
% whitening: from patch PSD
% flat-fielding: 1000
%

clear global;

global Npix cp;
global xd yd 
global SPV;
global xyz V_Fr V_Fi dummyX
global Rref
global imref;
global nfIm nfIm_F meanImage SDImage fPSD fPSD_patch inds_mask meanImage_orig SDImage_orig
global params

% % to instantiate a model object:
%#function model

% %% for params file:
% v_LSU=[-6.850777463139592 6.45811033010816 -7.110926673391666]';
% v_SSU=[-0.00995100158518693 0.005578154957851545 0.24269417166741894]';
% v_LSU=v_LSU./(params.aPerPix./10);
% v_SSU=v_SSU./(params.aPerPix./10);
% shifts=R_start*v_LSU + R_here*v_SSU;
% pred=cp-shifts([2 1])'; % % off by 4,1 @ j=127 (particle #3605) 

% function search_local
% aPerPix 1.07
% optimize_flag 1
% ff_flag 1
% subtract_flag 0
% subpixel_flag 1
% outputDir ~/smap_ij/patch/body_to_head_krios2_01
% % tableFile ~/smap_ij/table/part_6EK0_b0_SSU_body_krios2_032019_opt.mat
% modelFile ~/smap_ij/model/6EK0_b0_SSU_head_krios2.mrc
% v1 [3.5434 -8.3962 -3.9787]
% the_corr [-9.9750 0.7709 7.2623]
% the_q_corr [2.2810 -1.2900 1.0090 1.4849 5.2763 2.4149 1.9662 1.8082 1.1111]
% mask_rad 4.0
% rotationsFile ~/smap_ij/rotation/q_body_head_cyl_030620_01.txt
%
% patchFile 
% coordinateFile 
% nCores

%%

this_job_num=str2num(this_job_num)

params=smap.readParamsFile(paramsFile);
params
if(exist(params.outputDir,'dir')<7)
    try
        mkdir(params.outputDir);
    catch
    end;
end;
scratchDir=fullfile([params.outputDir '/scratch']);
if(exist(scratchDir,'dir')<7)
    try
        mkdir(scratchDir);
    catch
    end;
end;
fidLog=fopen(fullfile([scratchDir '/output_' smap.zp(this_job_num,4) '.log']),'w');
fprintf(fidLog,'%s\n',datestr(now,31));
fprintf(fidLog,'job %i of %i\n',this_job_num,params.nCores);
if( this_job_num==1 )
    copyfile(paramsFile,fullfile([params.outputDir '/']));
end;

Rref=smap.readRotationsFile(params.rotationsFile);
Rref=smap.normalizeRM(Rref);
params.q_offset=smap.normalizeRM(reshape(params.q_offset,3,3)');


try
    temp_var=load(params.tableFile);
    temp=fieldnames(temp_var);
    eval(['theTable=temp_var.' temp{1} ';']);
    fprintf(fidLog,'Working from table input: %s\n',char(params.tableFile));
    temp_patch=smap.mr(theTable(1,:).fn_patches,1,1);    
    Npix=size(temp_patch,1);
    cp=floor(Npix./2)+1;
catch
    % % 053121: loads patch/particle parameters from ASCII .par file instead of matlab table;
    % % needs full testing
    [~,resp]=system(['cat ' params.coordinateFile ' | awk ''(NR>1) {print $2,$3,$4,$9,$10,$11}''']);
    temp=split(resp); temp=temp(1:(end-1));
    temp=str2double(cellstr(temp));
    [xx,yy,zz]=deal(temp(1:6:end),temp(2:6:end),temp(3:6:end));
    R_patches=smap.normalizeRM(smap.frealign2smap([xx yy zz]));
    df_patches=[temp(4:6:end) temp(5:6:end) temp(6:6:end)];
    df_patches(:,1:2)=df_patches(:,1:2)./10;
    df_patches(:,3)=df_patches(:,3).*(-pi./180);
    [~,resp]=system(['cat ' strrep(params.coordinateFile,'.par','.txt') ' | awk ''(NR>1) {print $1,$2}''']);
    temp=split(resp); temp=temp(1:(end-1));
    temp=str2double(cellstr(temp));
    peaks=temp(2:2:end);
    
    part_varNames={'num_id', 'FOV_id', 'model_id', 'search_id', 'label', ...
        'peak', 'xy', 'q', 'df', 'T_min', 'xy_global', ...
        'df_image','xy_opt','q_opt','peak_opt'};
    temp_part=cell2table(cell(1,length(part_varNames)), 'VariableNames', part_varNames);

    fprintf(fidLog,'prepping table...\n');
    part=[];
    for j=1:length(peaks)
        temp_part.num_id=j;
        temp_part.FOV_id=nan;
        temp_part.model_id=nan;
        temp_part.search_id=nan;
        temp_part.label=nan;
        temp_part.peak=nan;
        temp_part.xy=[nan nan];%outVals(j,2:3);
        temp_part.q={nan(3,3)};
        temp_part.df=df_patches(j,:);
        temp_part.T_min=nan;
        temp_part.xy_global=[nan nan]; % fix
        temp_part.df_image=nan;
        temp_part.xy_opt=[cp cp];
        temp_part.q_opt={R_patches(:,:,j)};
        temp_part.peak_opt=peaks(j);
        part=[part; temp_part];
    end;
    theTable=part;
    
end;

inds_keep=[1:size(theTable,1)];
if( ~isempty(params.thr) )
    inds_keep=find(theTable.peak_opt>params.thr);
    theTable=theTable(inds_keep,:);
end;
nParticles=size(theTable,1);

% inds_job=smap.assignJobs(nParticles,params.nCores,this_job_num);
inds_job=[this_job_num:params.nCores:nParticles];
theTable=theTable(inds_job,:)
% R_patches=R_patches(:,:,inds_job);
% df_patches=df_patches(inds_job,:);
% peaks=peaks(inds_job);
nParticles=length(inds_job);


[~,fn_SPV,ext]=fileparts(params.modelFile)
fn_out=fullfile([scratchDir '/' fn_SPV '_cs_' smap.zp(this_job_num,4) '_table.mat'])



SPV=smap.mr(params.modelFile);
SPV=SPV-mode(SPV(:));

SPV=smap.cropOrPad(SPV,Npix.*[1,1,1]);
% rr_for_sinc=smap.rrj(ones(Npix,Npix,Npix,'single'));
% rs_sinc=sinc(double(rr_for_sinc)).^2;
% clear rr_for_sinc;
% SPV=SPV./rs_sinc;

[x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
cp=floor(Npix./2)+1;
x0=x(:,:,cp)-cp;
y0=y(:,:,cp)-cp;
z0=zeros(Npix,Npix);

xyz=[x0(:) y0(:) z0(:)];
xyz_temp=[x0(:) y0(:) z0(:)];
xyz=xyz_temp;
dummyX=[1:Npix];

clear x y z x0 y0 z0 xyz_temp;
V_F=fftshift(fftn(ifftshift(SPV)));
V_F(cp,cp,cp)=0;

V_Fr=single(real(V_F));
V_Fi=single(imag(V_F));
clear V_F SPV

mask=smap.rrj(ones(Npix,Npix)).*Npix;
inds_mask=find(mask(:)>(Npix/2));
clear mask;

ctr=1;
tableref_new=[];
SDs=[]; os_all=[]
cc_patch_all=[];
cc_patch_all_r=[];
new_patches=[];
for i=1:length(inds_job)
    tic;
    ind_here=inds_keep(inds_job(i));
%     nfIm=smap.mr(params.patchFile,ind_here,1);
    tableref=theTable(i,:);
    nfIm=smap.mr(char(tableref.fn_patches),tableref.patches_pageref,1);
    
    [tableref,cc_patch,cc_patch_r,RR]=constrained_search(tableref);
    if( params.optimize_flag==1 )
        [tableref,output_struct]=optimize_xcorr(tableref);
        [tableref,output_struct_ctrl]=optimize_xcorr(tableref,'ctrl');
        disp([num2str(tableref.peak_part) ' ' num2str(tableref.peak_part_opt) ' ' num2str(ctr)]);
    else
        disp([num2str(tableref.peak_part) ' ' num2str(ctr)]);
    end;
    if( params.new_patches_flag )
        new_patches=cat(3,new_patches,output_struct.new_patch);
        smap.mw(single(new_patches),strrep(fn_out,'_table.mat','_new.mrc'),params.aPerPix);
    end;
    tableref.model_id=string(fn_SPV);
    tableref_new=[tableref_new; tableref]
    cc_patch_all=cat(3,cc_patch_all,cc_patch);
    cc_patch_all_r=cat(3,cc_patch_all_r,cc_patch_r);
    ctr=ctr+1;
    save(fn_out,'tableref_new');
    smap.mw(single(cc_patch_all),strrep(fn_out,'_table.mat','_vals.mrc'),params.aPerPix);
    smap.mw(single(cc_patch_all_r),strrep(fn_out,'_table.mat','_ctrl.mrc'),params.aPerPix);
    the_interval=toc;
    fprintf(fidLog,'Done with patch %i of %i (elapsed: %5.2f sec.)\n',i,length(inds_job),the_interval);         
end;

% q_test=permute(reshape(cell2mat(tableref_new.q)',3,3,size(tableref_new,1)),[2 1 3]);

fn_done=strrep(fn_out,'_table.mat','_done.txt');
fid=fopen(fn_done,'w');
pause(1);
fclose(fid);
pause(1);

if( this_job_num < params.nCores )    
    fclose(fidLog);
    pause(1);
    return;
end;

%% cleanup:

fileTypesExpected={'table','vals','ctrl','done'};%,'log'};
if( params.new_patches_flag )
    fileTypesExpected=[fileTypesExpected {'new'}];
end;

while 1
    % % get a list of existing files to combine and do sanity check:
    nFilesExpected=params.nCores;
    fprintf(fidLog,'Looking for files from %i searches...\n',nFilesExpected);
    
    numFound=[]; fileTypesFound={};
    A=dir(fullfile([scratchDir '/*_cs_*.*']));
    ctr=1;
    for i=1:length(A)
        tempNum=regexp(A(i).name,'cs_(\d{4,4})_','tokens');
        if( length(tempNum)>0 )
            numFound(ctr)=str2num(char(tempNum{1}));
            tempType=regexp(A(i).name,'s_(\d{4,4})_','split');
            tempTypeParts=regexp(tempType{2},'(\.{1,1})','split');
            fileTypesFound{ctr}=tempTypeParts{1};
            ctr=ctr+1;
        end;
    end;
    
    if( ctr>(nFilesExpected.*length(fileTypesExpected)) )
        fprintf(fidLog,'combining %i files...\n',(nFilesExpected.*length(fileTypesExpected)));
        if( params.nCores>1 )
            pause(1);
        end;
        break;
    end;
    pause(1);
end;

inds={};
for i=1:length(fileTypesExpected)
    inds{i}=find(strcmp(fileTypesFound,fileTypesExpected{i})==1);
end;

theTable=[];
patches_all=[];
patches_all_ctrl=[];
new_patches=[];
% % read in each file-type and combine:
for i=1:length(inds{1})
    for j=1:length(fileTypesExpected)
        fileType=fileTypesExpected{j};
        fn=fullfile([scratchDir '/' A(inds{j}(i)).name]);
        switch fileType
            case 'table'
                try
                    load(fn,'tableref_new');
                    theTable=[theTable; tableref_new];
                catch
                    fprintf('skipped file %i\n',i);
                end
            case 'vals'
                z=smap.mr(fn);
                patches_all=cat(3,patches_all,z);
            case 'ctrl'
                z=smap.mr(fn);
                patches_all_ctrl=cat(3,patches_all_ctrl,z);
            case 'new'
                z=smap.mr(fn);
                new_patches=cat(3,new_patches,z);
            otherwise
%                 fprintf(fidLog,'file type %s not recognized...\n',fileType);
        end;
    end;
end;

[~,sI]=sort(theTable.num_id,'ascend');
theTable_filt=theTable(sI,:);
theTable=theTable_filt;
save(fullfile([params.outputDir '/patches.mat']),'theTable');

fn_patches=fullfile([params.outputDir '/patches.mrc']);
patches_all=patches_all(:,:,sI);
smap.mw(single(patches_all),fn_patches,params.aPerPix);
clear patches_all;

fn_patches_ctrl=fullfile([params.outputDir '/patches_ctrl.mrc']);
patches_all_ctrl=patches_all_ctrl(:,:,sI);
smap.mw(single(patches_all_ctrl),fn_patches_ctrl,params.aPerPix);
clear patches_all_ctrl;

if( params.new_patches_flag )
    fn_table=fullfile([params.outputDir '/particles_new.mat']);
    fn_patches_new=fullfile([params.outputDir '/particles_new.mrc']);
    new_patches=new_patches(:,:,sI);
    theTable_new=[];
    for i=1:size(theTable,1)
        tableref=theTable(i,:);
        xy_part=tableref.xy_opt+tableref.xy_part_opt-cp;
        tableref.xy_opt=xy_part;
        tableref.q_opt{1}=tableref.q_part_opt{1};
        tableref.peak_opt=tableref.peak_part_opt;
        tableref.peak=tableref.peak_part;
        temp=tableref(:,1:16); % % awkward
        theTable_new=[theTable_new; temp];
    end;
    theTable_new.fn_patches=repmat(string(fn_patches_new),size(theTable_new,1),1);
    theTable_new.patches_pageref=[1:size(theTable_new,1)]';    
    save(fn_table,'theTable_new');
    smap.mw(single(new_patches),fn_patches_new,params.aPerPix);
end;

fprintf(fidLog,'Done combining search output at %s\n',datestr(now,31));
fprintf(fidLog,'Consolidating log files...\n');

fclose(fidLog);
fnFinal=fullfile([params.outputDir '/search.log']);
fidFinal=fopen(fnFinal,'w');

tline={''};
for i=1:this_job_num
    fn=fullfile([scratchDir '/output_' smap.zp(i,4) '.log']);
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

        
try
    if( params.keep_scratch_flag )
        fprintf(fidFinal,'Keeping scratch files...\n');
    else
        fprintf(fidFinal,'Deleting scratch files...\n');
        delete(fullfile([scratchDir '/*.*']));
        rmdir(scratchDir);
    end;
end;

pause(1);
fclose(fidFinal);
pause(1);

return;

%%
function [tableref,cc_patch,cc_patch_r,RR]=constrained_search(tableref);

%%
global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global Rref
global imref
global nfIm meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new meanImage_orig SDImage_orig
global const inds_mask bgVal
global params
const=1i.*2.*pi;

% % make grid coordinates to use while applying phase shifts. Patch-sized
% % coordinates are used in the tight loop that does optimization:
[xd,yd]=meshgrid(((0:Npix-1)-Npix/2)/Npix,((0:Npix-1)-Npix/2)/Npix);

df_here=tableref.df;

% [Cs,Cc,V_acc,deltaE,a_i]=deal(params.Cs,params.Cc,params.V_acc, ...
%     params.deltaE,params.a_i);
params.Cs=tableref.search_params.Cs;
params.Cc=tableref.search_params.Cc;
params.V_acc=tableref.search_params.V_acc;
params.deltaE=tableref.search_params.deltaE;
params.a_i=tableref.search_params.a_i;
params.aPerPix=tableref.search_params.aPerPix;
CTF=ifftshift(smap.ctf(tableref.df,Npix.*[1,1]));

xy_LSU=[cp cp];%tableref.xy_opt;
R_start=smap.normalizeRM(tableref.q_opt{1});

% imref=smap.cutj(nfIm,Npix.*[1,1],xy_LSU);

% if( params.subtract_flag )
%     load(['~/smap_ij/model/' tableref(1,:).model_id{1} '.mat'])
%     tt=smap.templates(s,tableref(1,:).q_opt{1},tableref(1,:).df);
%     tt=smap.cutj(tt,Npix.*[1,1]);
%     tt=(tt-mean(tt(:)))./std(tt(:));
%     imref_temp=(imref-mean(imref(:)))./std(imref(:));
%     sf=(dot(imref_temp(:),tt(:))./dot(imref_temp(:),imref_temp(:)))
%     imref=imref_temp-tt.*sf;
% end;

[fPSD_patch,imref]=smap.psdFilter(nfIm-mean(nfIm(:)),'sqrt');
fPSD_patch(cp,cp)=0;
imref_F=smap.ftj(smap.nm(imref));

v1=(params.v1');%./(params.aPerPix./10);
the_q_corr=[params.the_q_corr(1:3); params.the_q_corr(4:6); params.the_q_corr(7:9)];
the_corr=(params.the_corr');%./(params.aPerPix./10);
nShifts=size(v1,2);

temp=smap.rrj(ones(Npix,Npix)).*Npix;
inds_cc=find(temp<=params.mask_rad);
temp=temp.*0;
temp(inds_cc)=1;
mask=temp;
[tempX,tempY]=meshgrid(1:Npix,1:Npix);
tempX=tempX-cp; tempY=tempY-cp;
inds_patch=find(abs(tempX)<=(params.patch_edge./2) & abs(tempY)<=(params.patch_edge./2));
clear tempX tempY;

nRotations=size(Rref,3);
% cc_patch=zeros(length(inds_cc).*nShifts,nRotations);
% cc_patch_r=zeros(length(inds_cc).*nShifts,nRotations);
cc_patch=zeros(length(inds_patch).*nShifts,nRotations);
cc_patch_r=zeros(length(inds_patch).*nShifts,nRotations);

if( params.ff_flag )

    nRotations_sample=1000;
    rng(1);
    q_rand=quaternion.randRot(1,nRotations_sample);
    R_rand=smap.normalizeRM(squeeze(RotationMatrix(q_rand)));
    
    sImage=zeros(Npix,Npix); ssImage=zeros(Npix,Npix);
    for i=1:nRotations_sample
        R_here=R_rand(:,:,i);
        xyz_r=(R_here'*xyz')';
        X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
        temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0);
        temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0);
        output_image=complex(temp_r,temp_i);
        
        projPot=reshape(output_image,Npix,Npix);
        ew=exp(1i*ifftn(ifftshift(projPot)));
        w_det=fftshift(ifftn(fftn(ew).*CTF));
        template=real(w_det.*conj(w_det));
        bgVal=nanmedian(template(inds_mask));
        template=template-bgVal;
        
        template_F=smap.ftj(template).*fPSD_patch;
        template_F=template_F./std(template_F(:));
        cc=smap.iftj(imref_F.*conj(template_F));
        sImage=sImage+cc;
        ssImage=ssImage+(cc.^2);
        if( mod(i,100)==0 ), fprintf('%i\n',i), end;
        
    end;
    meanImage=sImage./nRotations_sample;%double(mean(cc,3));
    squaredMeanImage=ssImage./nRotations_sample;%double(mean(cc.^2,3));
    SDImage=(sqrt(squaredMeanImage-(meanImage.^2)));
else
    SDImage=ones(Npix,Npix); meanImage=zeros(Npix,Npix);
end;
meanImage_orig=meanImage; SDImage_orig=SDImage;



tic

for i=1:nRotations    
    RR(:,:,i)=R_start*Rref(:,:,i);
end;

cc_best=[]; cc_best_r=[]; %tt=zeros(Npix,Npix,nRotations);
for i=1:nRotations
    R_here=RR(:,:,i);
    xyz_r=(R_here'*xyz')';
    X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
    temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0);
    temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0);
    output_image=complex(temp_r,temp_i);
    
    shifts=R_start*(v1+(the_q_corr*rotationMatrixToVector(inv(R_start)*R_here)')+the_corr);    
%     shifts=R_start*(v1);    
    
    shifts=shifts([2 1],:)./(params.aPerPix./10);
    pred=cp-shifts';
    
    projPot=reshape(output_image,Npix,Npix);
    ew=exp(1i*ifftn(ifftshift(projPot)));
    w_det=fftshift(ifftn(fftn(ew).*CTF));
    template=real(w_det.*conj(w_det));
    bgVal=nanmedian(template(inds_mask));
    template=template-bgVal;
%     smap.mw(single(template),fullfile([params.outputDir '/template_' smap.zp(tableref.num_id,3) '.mrc']),params.aPerPix);

    template_F=smap.ftj(template).*fPSD_patch;
    template_F=template_F./std(template_F(:));
    cc=smap.iftj(imref_F.*conj(template_F));
    cc=(cc-meanImage)./SDImage;
    cc_r=smap.iftj(imref_F.*conj(smap.rot90j(template_F,1)));
    cc_r=(cc_r-meanImage)./SDImage;
    cc_patch_temp=[]; cc_r_patch_temp=[];
    

%     % % for ctrl:
%     theShifts=theShifts+10 
    
    
%     theShifts=round(cp-pred);
%     cc_s=circshift(cc,theShifts); % this is going to be slow
%     cc_r_s=circshift(cc_r,theShifts); % this is going to be slow

    theShifts=cp-pred;
    if( params.subpixel_flag==0 )
        theShifts=round(fliplr(theShifts));
    end;
    
    cc_s=smap.applyPhaseShifts(cc,fliplr(theShifts));
    cc_r_s=smap.applyPhaseShifts(cc_r,fliplr(theShifts));
    
    temp=cc_s(inds_cc);
    cc_peak(i)=max(temp(:));
    cc_patch(:,i)=cc_s(inds_patch);%cc_patch_temp;
%     cc_peak(i)=max(cc_patch(:,i));
    
    temp=cc_r_s(inds_cc);
    cc_peak_r(i)=max(temp(:));
    cc_patch_r(:,i)=cc_r_s(inds_patch);%cc_r_patch_temp;
%     cc_peak_r(i)=max(cc_patch_r(:,i));
    
    if( cc_peak(i)==max(cc_peak) )
        cc_best=cc_s.*mask;
%         cc_best=circshift(cc_best,-theShifts);
        cc_best=smap.applyPhaseShifts(cc_best,-fliplr(theShifts));
        ind_best=i;
    end;
    if( cc_peak_r(i)==max(cc_peak_r) )
        cc_best_r=cc_r_s.*mask;
%         cc_best_r=circshift(cc_best_r,-theShifts);
        cc_best_r=smap.applyPhaseShifts(cc_best_r,-fliplr(theShifts));
        ind_best_r=i;
    end;
    if( mod(i,100)==0 ), fprintf('%i\t%6.2f\n',i,max(cc_patch(:))), end;
    
end

[nx,ny]=find(cc_best==max(cc_best(:)),1,'first');
xy_part=round(xy_LSU+([nx ny]-cp));
tableref.xy_part=xy_part;
tableref.q_part{1}=RR(:,:,ind_best);
tableref.peak_part=cc_peak(ind_best);
% tableref.cc_patch={cc_patch};
% tableref.RR={RR};

[nx,ny]=find(cc_best_r==max(cc_best_r(:)),1,'first');
xy_part_r=round(xy_LSU+([nx ny]-cp));
tableref.xy_part_ctrl=xy_part_r;
tableref.q_part_ctrl{1}=RR(:,:,ind_best_r);
tableref.peak_part_ctrl=cc_peak_r(ind_best_r);
tableref_out=tableref;
toc
% ack

%%

function [tableref,output_struct]=optimize_xcorr(tableref,varargin);

%%
global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global R_here
global imref
global nfIm nfIm_F meanImage SDImage fPSD fPSD_patch meanref SDref CTF shifts_new meanImage_orig SDImage_orig
global const inds_mask bgVal
global ctrl_flag params
const=1i.*2.*pi;

% meanImage=meanImage_orig;
% SDImage=SDImage_orig;

ctrl_flag=0;
if( nargin>1 )
    ctrl_flag=strcmp(char(varargin{1}),'ctrl')
end;
df_here=tableref.df;

if( ctrl_flag )
    xy_here=tableref.xy_part_ctrl; % %
    R_here=smap.normalizeRM(tableref.q_part_ctrl{1}');
else
    xy_here=tableref.xy_part; % %
    R_here=smap.normalizeRM(tableref.q_part{1}');
end;

[fPSD_patch,imref]=smap.psdFilter(nfIm-mean(nfIm(:)),'sqrt');
fPSD_patch(cp,cp)=0;
imref=smap.nm(imref);
% imref=smap.cutj(imref,Npix.*[1,1],xy_here);
% meanImage=smap.cutj(meanImage_orig,Npix.*[1,1],xy_here);
% SDImage=smap.cutj(SDImage_orig,Npix.*[1,1],xy_here);
imref=smap.applyPhaseShifts(imref,cp-fliplr(xy_here));
meanImage=smap.applyPhaseShifts(meanImage_orig,cp-fliplr(xy_here));
SDImage=smap.applyPhaseShifts(SDImage_orig,cp-fliplr(xy_here));

% [fPSD_patch,imref]=smap.psdFilter(imref-mean(imref(:)),'sqrt');
% fPSD_patch(cp,cp)=0;
imref_F=smap.ftj(imref);

% % start by centering the peak:
xyz_r=(R_here*xyz')';
X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0);
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0);
output_image=complex(temp_r,temp_i);

projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
bgVal=nanmedian(template(inds_mask));
template=template-bgVal;
template_F=smap.ftj(template).*fPSD_patch;
if( ctrl_flag )
    template_F=smap.rot90j(template_F,1);
end;
template_F=template_F./std(template_F(:));
cc=smap.iftj(imref_F.*conj(template_F));
cc_corr=(cc-meanImage)./SDImage;
cc_F=smap.ftj(cc_corr);
% cc_F=imref_F.*conj(template_F);

options = optimset('Display','off','TolFun',1e-3);

ff=@(x)-optimize_phase_shifts(cc_F,x);
[shifts,fval]=fminsearch(ff,[0 0],options);
xy_new=(xy_here-shifts);
if( ctrl_flag )
    tableref.xy_part_opt_ctrl=xy_new;%(xy_here-shifts);
else
    tableref.xy_part_opt=xy_new;%(xy_here-shifts);
end

[fPSD_patch,imref]=smap.psdFilter(nfIm-mean(nfIm(:)),'sqrt');
fPSD_patch(cp,cp)=0;
imref=smap.nm(imref);
% imref=smap.cutj(imref,Npix.*[1,1],xy_new);
% meanImage=smap.cutj(meanImage_orig,Npix.*[1,1],xy_new);
% SDImage=smap.cutj(SDImage_orig,Npix.*[1,1],xy_new);
imref=smap.applyPhaseShifts(imref,cp-fliplr(xy_new));
meanImage=smap.applyPhaseShifts(meanImage_orig,cp-fliplr(xy_new));
SDImage=smap.applyPhaseShifts(SDImage_orig,cp-fliplr(xy_new));
imref_F=smap.ftj(imref);


% x0=[0 0 0 0 0];
x0=[1 0 0 0 0 0];
f=@(x)-optimize_angle(imref_F,x);
[angles,fval,exitflag,output_struct]=fminsearch(f,x0,options);
RM=smap.q2R(angles(1:4))*R_here;
shifts_new=angles(5:6).*100;
if( ctrl_flag )
    tableref.q_part_opt_ctrl{1}=RM';
else
    tableref.q_part_opt{1}=RM';
end;

%shifts=shifts+shifts_new;
%xy_new=(xy_here-shifts);
xy_new=xy_new-shifts_new;
if( ctrl_flag )
    tableref.xy_part_opt_ctrl=xy_new;%(xy_here-shifts);
else
    tableref.xy_part_opt=xy_new;%(xy_here-shifts);
end;


xyz_r=(RM*xyz')';
X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0);
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0);
output_image=complex(temp_r,temp_i);
projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
template=template-bgVal;

% % for patch values:
[fPSD_patch,imref]=smap.psdFilter(nfIm-mean(nfIm(:)),'sqrt');
fPSD_patch(cp,cp)=0;
imref=smap.nm(imref);
% imref=smap.cutj(imref,Npix.*[1,1],xy_new);
% meanImage=smap.cutj(meanImage_orig,Npix.*[1,1],xy_new);
% SDImage=smap.cutj(SDImage_orig,Npix.*[1,1],xy_new);
imref=smap.applyPhaseShifts(imref,cp-fliplr(xy_new));
meanImage=smap.applyPhaseShifts(meanImage_orig,cp-fliplr(xy_new));
SDImage=smap.applyPhaseShifts(SDImage_orig,cp-fliplr(xy_new));
imref_F=smap.ftj(imref);


% imref_s=smap.cutj(nfIm,Npix.*[1,1],xy_new);
% imref_F=smap.ftj(imref_s).*fPSD_patch;
% meanImage=smap.cutj(meanImage_orig,Npix.*[1,1],xy_new);
% SDImage=smap.cutj(SDImage_orig,Npix.*[1,1],xy_new);
% meanImage=smap.applyPhaseShifts(meanImage,fliplr(shifts_new));
% SDImage=smap.applyPhaseShifts(SDImage,fliplr(shifts_new));

% meanref=smap.cutj(meanImage,Npix.*[1,1],xy_new);
% SDref=smap.cutj(SDImage,Npix.*[1,1],xy_new);

template_F=smap.ftj(template).*fPSD_patch;
template_F=template_F./std(template_F(:));
if( ctrl_flag )
    template_F=smap.rot90j(template_F,1);
end;

cc=smap.iftj(imref_F.*conj(template_F));
cc=(cc-meanImage)./SDImage;
% temp=smap.cutj(cc,[5,5],[cp,cp]);

if( ctrl_flag )
    %tableref.peak_part_opt_ctrl=gather(cc_corr(xy_here(1),xy_here(2)));
    tableref.peak_part_opt_ctrl=cc(cp,cp);%(temp(3,3));
else
    %tableref.peak_part_opt=gather(cc_corr(xy_here(1),xy_here(2)));
    tableref.peak_part_opt=cc(cp,cp);%(temp(3,3));
    if( params.new_patches_flag )
        temp=smap.applyPhaseShifts(nfIm,fliplr(cp-xy_new));
        output_struct.new_patch=temp;
    end;
end;

% imsc(cc_corr,[tableref.xy_part],10); title(num2str(tableref.peak_part_opt)); drawnow;

%%
function outref=optimize_angle(imref_F,angles);

global Npix cp;
global xd yd
global xyz V_Fr V_Fi dummyX
global R_here
global meanref SDref fPSD_patch CTF meanImage SDImage
global const bgVal
global ctrl_flag 

shifts=angles(5:6).*100;

%RM=RotationMatrix(quaternion.eulerangles('xyz',angles(1:3).*pi./180))*Rref;
RM=smap.q2R(angles(1:4))*R_here;

xyz_r=(RM*xyz')';
X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
temp_r = interpn(dummyX,dummyX,dummyX,V_Fr,Y,X,Z,'linear',0);
temp_i = interpn(dummyX,dummyX,dummyX,V_Fi,Y,X,Z,'linear',0);
output_image=complex(temp_r,temp_i);
projPot=reshape(output_image,Npix,Npix);
ew=exp(1i*ifftn(ifftshift(projPot)));
w_det=fftshift(ifftn(fftn(ew).*CTF));
template=real(w_det.*conj(w_det));
template=template-bgVal;%median(template(:));
template_F=smap.ftj(template).*fPSD_patch;
template_F=template_F./std(template_F(:));
if( ctrl_flag )
    template_F=smap.rot90j(template_F,1);
end;

temp=smap.iftj(imref_F.*conj(template_F));
temp=(temp-meanImage)./SDImage;
cc_F=(smap.ftj(temp));

%cc_F=imref_F.*conj(template_F);

dphs=xd'.*(-shifts(1))+yd'.*(-shifts(2));
dphs=exp(const.*dphs);
d_done=cc_F.*dphs;
d_done=real(ifftn(ifftshift(d_done)));
outref=double(gather(d_done(1)).*Npix);


%%
function y=optimize_phase_shifts(patchref,shifts)
% % mod 060819

global xd yd const Npix;

dphs=xd'.*(-shifts(1))+yd'.*(-shifts(2));
dphs=exp(const.*dphs);
d_done=patchref.*dphs;
d_done=real(ifftn(ifftshift(d_done)));
y=double(gather(d_done(1).*Npix));%.*Npix;



%%
function rotationVector = rotationMatrixToVector(rotationMatrix)
rotationVector=rodriguesMatrixToVector(rotationMatrix')';

%%
function [rotationVector, dvdM] = rodriguesMatrixToVector(rotationMatrix)
% rodriguesMatrixToVector Convert a 3D rotation matrix into a rotation
% vector.
%
% rotationVector = rodriguesMatrixToVector(rotationMatrix) computes a 
% rotation vector (axis-angle representation) corresponding to a 3D 
% rotation matrix using the Rodrigues formula.
%
% rotationMatrix is a 3x3 3D rotation matrix.
%
% rotationVector is a 3-element rotation vector corresponding to the 
% rotationMatrix. The vector represents the axis of rotation in 3D, and its 
% magnitude is the rotation angle in radians.
%
% [..., dvdM] = rodriguesMatrixToVector(rotationMatrix) additionally
% returns the 3-by-9 derivatives of rotationVector w.r.t rotationMatrix.
%
% See also vision.internal.calibration.rodriguesVectorToMatrix

% References:
% [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer
%     Vision," Cambridge University Press, 2003.
% 
% [2] E. Trucco, A. Verri. "Introductory Techniques for 3-D Computer
%     Vision," Prentice Hall, 1998.

% Copyright 2012 MathWorks, Inc.

%#codegen

% get the rotation matrix that is the closest approximation to the input
[U, ~, V] = svd(rotationMatrix);
rotationMatrix = U * V';

t = trace(rotationMatrix);
% t is the sum of the eigenvalues of the rotationMatrix.
% The eigenvalues are 1, cos(theta) + i sin(theta), cos(theta) - i sin(theta)
% t = 1 + 2 cos(theta), -1 <= t <= 3

tr = (t - 1) / 2;
theta = real(acos(tr));

r = [rotationMatrix(3,2) - rotationMatrix(2,3); ...
     rotationMatrix(1,3) - rotationMatrix(3,1); ...
     rotationMatrix(2,1) - rotationMatrix(1,2)];

needJacobian = nargout > 1;
outType = class(rotationMatrix);

if needJacobian
    dtrdM = [1 0 0 0 1 0 0 0 1] / 2;
    dtrdM = cast(dtrdM, outType);
    
    drdM = [0 0 0 0 0 1 0 -1 0;...
            0 0 -1 0 0 0 1 0 0;...
            0 1 0 -1 0 0 0 0 0];
    drdM = cast(drdM, outType);
end

threshold = cast(1e-4, outType); 

if sin(theta) >= threshold
    % theta is not close to 0 or pi
    vth = 1 / (2*sin(theta));
    v = r * vth;
    rotationVector = theta * v;

    if needJacobian        
        dthetadtr = -1/sqrt(1-tr^2);
        dthetadM = dthetadtr * dtrdM;
        
        % var1 = [vth; theta]
        dvthdtheta = -vth*cos(theta)/sin(theta);
        dvar1dtheta = [dvthdtheta; 1];
        dvar1dM =  dvar1dtheta * dthetadM;

        % var = [r;vth;theta]
        dvardM = [drdM;dvar1dM];

        % var2 = [r;theta]
        dvdvar = [vth*eye(3,outType) r zeros(3,1,outType)];
        dthetadvar = cast([0 0 0 0 1], outType);
        dvar2dvar = [dvdvar; dthetadvar];

        dVdvar2 = [theta*eye(3,outType) v];

        dvdM = dVdvar2 * dvar2dvar * dvardM;
    end
elseif t-1 > 0
    % theta is close to 0
    rotationVector = (.5 - (t - 3) / 12) * r;
    if needJacobian   
        dvdM = (-1 / 12) * r * dtrdM + (.5 - (t - 3) / 12) * drdM;
    end
else
    % theta is close to pi
    rotationVector = ...
        computeRotationVectorForAnglesCloseToPi(rotationMatrix, theta);
    if needJacobian
        % No definition for this case
        dvdM = zeros(3, 9, outType);
    end
end

%%
function rotationVector = computeRotationVectorForAnglesCloseToPi(rotationMatrix, theta)
% r = theta * v / |v|, where (w, v) is a unit quaternion.
% This formulation is derived by going from rotation matrix to unit
% quaternion to axis-angle

% choose the largest diagonal element to avoid a square root of a negative
% number
[~, a] = max(diag(rotationMatrix));
a = a(1);
b = mod(a, 3) + 1;
c = mod(a+1, 3) + 1;

% compute the axis vector
s = sqrt(rotationMatrix(a, a) - rotationMatrix(b, b) - rotationMatrix(c, c) + 1);
v = zeros(3, 1, 'like', rotationMatrix);
v(a) = s / 2;
v(b) = (rotationMatrix(b, a) + rotationMatrix(a, b)) / (2 * s);
v(c) = (rotationMatrix(c, a) + rotationMatrix(a, c)) / (2 * s);

rotationVector = theta * v / norm(v);



