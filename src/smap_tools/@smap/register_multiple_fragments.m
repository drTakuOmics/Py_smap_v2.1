function [outref,shifts]=register_multiple_fragments(fn_array,varargin);
%
% fn_array is a cell array of filenames. The first filename should be the
% combined model, and the rest should the indicate smaller fragments to be
% brought into alignment with the combined model (shifts only). Optional
% varargins are: 1) final box size for the models (in pixels; defaults to
% the box size of the combined model; and 2) max allowed shift (in pixels; 
% defaults to 200).
%
% 
% 
% function outref=register_multiple_fragments(fnArray,varargin);

%%

edgeSize=[];
maxShift=200;
if( nargin>1 )
    if( ~isempty(varargin{1}) )
        edgeSize=varargin{1};
    end;
    if( nargin>2 )
        if( ~isempty(varargin{2}) )
            maxShift=varargin{2};
        end;
    end;
end;

fn_ref=fn_array{1};
nModels=length(fn_array);
fn_out=[];
fn_out{1}=fn_ref;


fprintf('reading %s...',fn_ref);
[spv_ref,rez]=smap.mr(fn_ref);
fprintf('\n');

if( isempty(edgeSize) )
    edgeSize=size(spv_ref,1);
end;

bgVal_ref=mode(spv_ref(:));
if( size(spv_ref,1)<edgeSize )
    fprintf('extending reference edge size from %s to %s pixels...\n',num2str(size(spv_ref,1)),num2str(edgeSize));
    spv_ref=smap.extendj(spv_ref,edgeSize.*[1,1,1],bgVal_ref);
elseif( size(spv_ref,1)>edgeSize )
    fprintf('cropping reference edge size from %s to %s pixels...\n',num2str(size(spv_ref,1)),num2str(edgeSize));
    spv_ref=smap.cutj(spv_ref,edgeSize.*[1,1,1]);
else
    fprintf('leaving reference edge size at %s pixels\n',num2str(edgeSize));
end;

spv_reg=zeros(edgeSize,edgeSize,edgeSize,nModels,'single');
spv_reg(:,:,:,1)=spv_ref;
rez_new=[];
rez_new{1}=rez;
for i=2:nModels
    fn_new=fn_array{i};
    fn_out{i}=strrep(fn_array{i},'SP.mrc','sSP.mrc');
    fprintf('reading %s...',fn_new);
    % should allow scaling here (TODO)
    [spv_new,rez_new{i}]=smap.mr(fn_new);
    fprintf('\n');
    if( size(spv_new,1)<edgeSize )
        fprintf('extending fragment edge size from %s to %s pixels...\n',num2str(size(spv_new,1)),num2str(edgeSize));
        padVal_new=mode(spv_new(:));
        spv_new=smap.extendj(spv_new,edgeSize.*[1,1,1],padVal_new);
    elseif( size(spv_new,1)>edgeSize )
        fprintf('cropping fragment edge size from %s to %s pixels...\n',num2str(size(spv_new,1)),num2str(edgeSize));
        spv_new=smap.cutj(spv_new,edgeSize.*[1,1,1]);
    else
        fprintf('leaving fragment edge size at %s pixels\n',num2str(edgeSize));
    end;
    [spv_reg(:,:,:,i),shifts(i,:) ]=smap.reg2vols(spv_new,spv_ref,maxShift);
end;
outref=squeeze(sum(spv_reg(:,:,:,2:end),4));
bgVal=mode(outref(:));
outref=outref-bgVal+bgVal_ref;

imStack_ref=squeeze(sum(spv_ref,1));
imStack_ref=[imStack_ref squeeze(sum(spv_ref,2))];
imStack_ref=[imStack_ref squeeze(sum(spv_ref,3))];

imStack_reg=squeeze(sum(outref,1));
imStack_reg=[imStack_reg squeeze(sum(outref,2))];
imStack_reg=[imStack_reg squeeze(sum(outref,3))];


imStack=cat(1,imStack_ref,imStack_reg);
figure(404); clf;
imsc(imStack);

% put into PDB frame of reference (still units of pixels):
shifts=-shifts;
shifts(:,1:2)=fliplr(shifts(:,1:2));

for i=1:size(shifts,1)
    fprintf('%4.2f\t%4.2f\t%4.2f\n',shifts(i,1),shifts(i,2),shifts(i,3));
end;

%disp('paused'); pause;
% for i=2:nModels
%     fprintf('writing %s...\n',fn_out{i});
%     smap.mw(single(squeeze(spv_reg(:,:,:,i))),fn_out{i},rez_new{i});
% end;






