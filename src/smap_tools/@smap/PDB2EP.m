function obj=PDB2EP(obj);%structureName,nmPerVoxel,varargin);
% Calculates electrostatic potential from .pdb file using TEM-simulator 
projName='trash.mrc';

% % hard-coded:
% modelsDir=['/tier2/denk/models/'];
modelsDir=[smap.checkBaseDir 'models/'];
% modelsDir=[get(handles.userPrefsBox,'UserData') 'models/'];
% mapsDir=['~/matching/temp/'];
% if( nargin>2 )
% %     handles=varargin{1};
% %     mapsDir=handles.user.prefs.dirs.maps;
% %     mapsDir=smap.getPref(handles,'mapsDir');
%     modelsDir=[get(handles.userPrefsBox,'UserData') 'models/'];
% end;

nmPerVoxel=obj.prop.nmPerPixel_EP;
modelPDB=obj.ID.PDB;
modelName=obj.ID.ID;

pdbLine=['pdb_file_in = ' modelPDB '.pdb'];
voxelLine=['voxel_size = ' num2str(nmPerVoxel)];

df=70/1e3;
% logLine=['log_file = ' modelsDir modelPDB '/' modelName '_' datestr(now,30) '.log'];
logLine=['log_file = ' modelName '_' datestr(now,30) '.log'];
dfLine=strcat('defocus_nominal = ',[' ' sprintf('%4.3f',df)]);
imFileLine=['image_file_out = ' projName ];
mapFileLine=['map_file_re_out = ' modelName '_EP.mrc'];

coordFileLine=['coord_file_in = arb_location.txt'];
rotationFileLine=['geom_file_in = arb_rotations.txt'];

of={
    '=== simulation === '
    'generate_micrographs = yes'
    logLine
    ''
    '=== sample ==='
    'diameter = 300 # in nm; diameter of hole in carbon'
    'thickness_center = 0'
    'thickness_edge = 0'
    ''
    '=== particle arb ==='
    'source = pdb'
    pdbLine
    'add_hydrogen = no'
    voxelLine
    mapFileLine
    'map_file_im_out = arb_im_map.mrc'
    ''
    '=== particleset ==='
    'particle_type = arb'
    'particle_coords = file'
    coordFileLine
    ''
    '=== geometry ==='
    'gen_tilt_data = yes'
    'tilt_mode = single_particle'
    rotationFileLine
    'ntilts = 1'
    'theta_start = 0 '
    'theta_incr = 0 '
    'geom_errors = none'
    ''
    '=== electronbeam ==='
    'acc_voltage = 300 # in kV'
    'energy_spread = 0.0 # in eV'
    'gen_dose = yes'
    'dose_per_im = 1000 # e/nm^2'
    ''
    '=== optics ==='
    'magnification = 103626 # set to give final pixel size of 0.965A/pix (= 5e4 A/pix / (0.965A/pix))'
    'cs = 2.7 # in mm'
    'cc = 0 # in mm'
    'aperture = 300 # in mm; aperture in focal plane (back focal plane?)'
    'focal_length = 3.5 # in mm; focal length of ?primary lens? (objective, presumably)'
    %         'cond_ap_angle = 0.05 # in millirad (aperture angle of condenser lens)'
    'cond_ap_angle = 0.00 # in millirad (aperture angle of condenser lens)'
    'gen_defocus = yes'
    dfLine
    ''
    '=== detector ==='
    'det_pix_x = 512'
    'det_pix_y = 512'
    'padding = 150'
    'pixel_size = 5'
    'gain = 1'
    'use_quantization = no'
    'dqe = 1'
    'mtf_a = 0.5'
    'mtf_b = 0.5'
    'mtf_c = 0 '
    'mtf_alpha = 0'
    'mtf_beta = 0'
    imFileLine
    };



fnOut=[modelsDir modelPDB '/' modelName '_PDB2EP.txt'];
cd([modelsDir modelPDB]);
pwd
fid=fopen(fnOut,'w');
for i=1:length(of)
    fprintf(fid,'%s\n',of{i});
end;
fclose(fid);

fnOut=[modelsDir modelPDB '/arb_location.txt'];
fid=fopen(fnOut,'w');
fprintf(fid,'1  6\n');
fprintf(fid,'#\t x\t y\t z\t phi\t theta\t psi\n');
fprintf(fid,'\t %7.4f\t %7.4f\t %7.4f\t %7.4f\t %7.4f\t %7.4f\n',zeros(6,1));
fclose(fid);

fnOut=[modelsDir modelPDB '/arb_rotations.txt'];
fid=fopen(fnOut,'w');
fprintf(fid,'1  3\n');
fprintf(fid,'#\t phi\t theta\t psi\n');
fprintf(fid,'\t %7.4f\t %7.4f\t %7.4f\n',zeros(3,1));
fclose(fid);

disp(['Using TEM-simulator to calculate EP for ' modelName '...']);

words=['TEM-simulator ' modelName '_PDB2EP.txt'];
system(words);

% sampleIm=mr('trash.mrc');
% smapDir=smap.getPref(handles,'smapDir');
% cd(smapDir); cd('../');

% cd('/tier2/denk/');
cd([smap.checkBaseDir]);
disp(pwd);

obj.ID.EP=1;
obj.prop.EPName=[modelsDir modelPDB '/' modelName '_EP.mrc'];
s=obj;
% save(['/tier2/denk/models/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'models/' obj.ID.ID '.mat'],'s');

disp('done');

