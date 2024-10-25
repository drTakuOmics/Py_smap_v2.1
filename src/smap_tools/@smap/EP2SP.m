function obj=EP2SP(obj);%structureName,pixelSize,varargin);
% Calculates downsampled scattering potential from electrostatic potential
% 
fprintf('calculating scattering potential...\n');

% modelsDir=['/tier2/denk/models/'];
modelsDir=[smap.checkBaseDir 'models/'];

% mapsDir=['~/matching/temp/'];
% if( nargin>2 )
%     handles=varargin{1};
% %     mapsDir=handles.user.prefs.dirs.maps;
%     mapsDir=smap.getPref(handles,'mapsDir');
% end;

nmPerVoxel_EP=obj.prop.nmPerPixel_EP;
nmPerVoxel_SP=obj.prop.nmPerPixel_SP;
modelPDB=obj.ID.PDB;
modelName=obj.ID.ID;

outDir=[modelsDir modelPDB '/']
fname=[outDir modelName '_EP.mrc']

[pv,aPerPix_pv]=smap.mr(fname);
fprintf('Read in %s with %7.4f nm voxel size\n',fname,aPerPix_pv);

aPerPix_final=nmPerVoxel_SP.*10;%pixelSize.*10;

dx=aPerPix_final./1e10;

cc=smap.def_consts();

% convert into a cube:
mpv=min(pv(:));
pvMinDim=min(size(pv));
pvMaxDim=max(size(pv));
if( pvMinDim<pvMaxDim )
    disp('making a cube...');
    pv_cube=single(smap.extendj(pv,[pvMaxDim,pvMaxDim,pvMaxDim],mpv));
else
    pv_cube=pv;
end;
clear pv;

% subtract the background (water) potential:
bgVal=pv_cube(1,1,1);
pv_cube=pv_cube-bgVal;

% 120615:
disp('resizing...');
SPV=smap.resize_F(pv_cube,nmPerVoxel_EP/nmPerVoxel_SP);

pd=smap.particleDiameter(SPV,0.05);
edgeSize=2.5*pd

disp('padding...');
edgeSize=max([edgeSize max(size(SPV))]);
SPV_out=single(smap.extendj(SPV,[edgeSize,edgeSize,edgeSize],0));

disp('computing phase shifts...');
SPV_out_2=SPV_out.*cc.IC.*dx./(2.*cc.k);

% %
% % this seems like a mistake because it reverses the handedness (052416):
% %
% % and as of 080215 we are reversing the slice order, as before Rullgard:
temp=zeros(size(SPV_out_2));
for i=1:size(SPV_out_2,3)
    temp(:,:,i)=SPV_out_2(:,:,size(SPV_out_2,3)+1-i);
end;
SPV_out_3=temp;

SPV_out_3=smap.applyPhaseShifts(SPV_out_3,[0 0 1]);

% % write to files for matching:
% %
% % move this into a separate method
% %
SPV=single(SPV_out_3);
if( exist('spv_ref','var')==1 )
    disp('registering with reference volume...');
    spv_reg=reg2vols(SPV,spv_ref,200);
    for i=1:3
        imsc(squeeze(sum(spv_ref,i))); pause; imsc(squeeze(sum(spv_reg,i))); pause;
    end;
    SPV=spv_reg;
end;

for i=1:3
    outref{i}=squeeze(sum(SPV,i));
end;

% SPVName=[outDir 'SPV.mrc'];
SPVName=[outDir modelName '_SP.mrc']
obj.ID.SP=1;
obj.prop.SPName=SPVName;
smap.mw(single(SPV),SPVName,aPerPix_final);

s=obj;
% save(['/tier2/denk/models/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'models/' obj.ID.ID '.mat'],'s');


disp('done');


