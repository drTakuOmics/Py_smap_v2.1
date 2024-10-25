function [outref,scaled_template]=subtractVolume(recref,templateref,varargin);

subtraction_mode='normal';
if( nargin>2 )
    subtraction_mode=varargin{1};
end;

N=size(recref,1);
cp=floor(N./2)+1;

rr=smap.rrj(ones(N,N,N,'single')).*N;
mask=single(rr<(cp-1));

template=templateref-mode(templateref(:));
template_F=smap.ftj(template);
rec=recref-mean(recref(:));

rec_norm=smap.nm(rec);
template_norm=smap.nm(template);

%
% % this threshold is somewhat arbitrary:
%
% inds_norm=find(abs(template_norm(:))>0.01); % oversubtracted in sharp points
% inds_norm=find(abs(template_norm(:))>0.1);
inds_norm=find(abs(template_norm(:))>0.5);
% inds_norm=find(abs(template_norm(:))>1); % meh

temp=zeros(N,N,N);
temp(inds_norm)=rec(inds_norm);

rec_F=smap.ftj(temp);

fprintf('calculating radial averages...')
[a,q2]=smap.radialmeanj(abs(rec_F));
[a_t,q2_t]=smap.radialmeanj(abs(template_F));
fprintf('\n');

template_mod_F=template_F.*(q2./q2_t);

% temp_F=smap.ftj(rec);
% temp_F=temp_F-template_mod_F;
% outref=smap.iftj(temp_F);
% templateref_m=smap.iftj(template_mod_F);


scaled_template=smap.iftj(template_mod_F);
switch subtraction_mode
    case 'normal'
        outref=rec-scaled_template;
    case 'twofold'
        outref=rec-2.*scaled_template;
    otherwise
        fprintf('Subtraction mode not recognized (options are normal or twofold)...doing normal subtraction\n');
        outref=rec-scaled_template;
end;

