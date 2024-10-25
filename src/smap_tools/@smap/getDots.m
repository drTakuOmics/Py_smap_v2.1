function obj_out=getDots(obj,varargin);

baseDir=smap.checkBaseDir;
cd(baseDir);
objectPath=[baseDir 'searches/' obj.ID.ID '.mat']
 
disp(obj);

load rotations/rotations_hopf_R.mat

seedWords={'mt19937ar','Seed',eval('1')};
stream = RandStream(seedWords{1},seedWords{2},seedWords{3});
RandStream.setGlobalStream(stream);
randInds=randperm(size(R,3));
randInds=randInds(1:1e3);
%randInds(1)=1:100;
%randInds=[1:1e2];

df=[obj.params.df1 obj.params.df2 -obj.params.ast];
V=smap.ri(obj.search_objects.model.prop.SPName);
obj.params.maskSize=smap.particleDiameter(V).*1.25;
obj.params.maskSize=min([obj.params.maskSize min(size(V))]);
disp(obj.params.maskSize);
obj.params.edgeSize=min([obj.params.edgeSize min(size(V))])
R_forMask=single(smap.rrj(zeros(obj.params.edgeSize,obj.params.edgeSize))).*(floor(obj.params.edgeSize./2)./0.5);
rMask=ones(size(R_forMask));
rMask(find(R_forMask>obj.params.maskSize/2))=nan;

edgeSize=obj.params.edgeSize;
tt=smap.templates_gpu(obj.search_objects.model,R(:,:,randInds),df,[],edgeSize);
%tt=smap.cutj(tt,[obj.params.edgeSize,obj.params.edgeSize,size(tt,3)]);

bgVal=median(tt(:));
for i=1:size(tt,3)
    template=tt(:,:,i);
    template=template-bgVal;
    % normalize and compute the dot product for classification:
    tMasked=template.*rMask;
    temp_dots(i)=nanstd(tMasked(:));
end;

%disp('paused'); pause; 

md=mean(temp_dots);
sd=std(temp_dots);

md=md.*1.125; % hack; reason TBD

disp(md); disp(sd);

obj.params.dot_mean=md;
obj.params.dot_std=sd;

s=obj;
obj_out=s;
save(objectPath,'s');
