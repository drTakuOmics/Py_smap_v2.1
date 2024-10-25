function outref=resize_F(inref,sf,varargin);
% Fourier crop or pad to a new size; use 'newSize' to scale boundaries, too
% skip varargin to keep the original size;
% 
% outref=resize_F(inref,sf,varargin);
% sf = pitch_orig/pitch_target
%

method='fixedSize';
if( nargin>2 )
    method=varargin{1};
end;

sfAmp=abs(1-sf);
if( sfAmp>0.2 )
    sfType='large';
else
    sfType='small';
end;

os=size(inref);
finalSize=floor(os.*sf);
bgVal=mode(inref(:));
% inref=inref-bgVal;
inref_F=fftshift(fftn(ifftshift(inref-bgVal)));

if( sf>1 ) % Fourier crop (downsample/zoom in):
    os=[size(inref,1) size(inref,2) size(inref,3)];
%     bgVal=mode(inref(:));
    finalSize=floor([size(inref,1) size(inref,2)].*sf);
    inref_F=fftshift(fftn(ifftshift(inref)));
    if( size(inref,3)==1 )
%         fprintf('downsampling...\n');
        inref_F_padded=smap.cropOrPad(inref_F,[finalSize(1),finalSize(end)],0);
        outref=real(fftshift(ifftn(ifftshift(inref_F_padded))));
        outref=outref.*(sf).^2;

        switch method
            case 'fixedSize'
                outref=smap.cutj(outref,[os(1),os(2)]);
            case 'newSize'
                outref=outref;
            otherwise
                outref=outref;
        end;
    else
        inref_F_padded=smap.cropOrPad(inref_F,[finalSize,finalSize,finalSize],0);
        outref=real(fftshift(ifftn(ifftshift(inref_F_padded))));
        outref=outref.*(1./sf).^3;
%        
        if( strcmp(method,'fixedSize') )
            if( size(outref,1)<os(1) )
                outref=smap.cropOrPad(outref,os,0);
            elseif( size(outref,1)>os(1) )
                outref=smap.cropOrPad(outref,os);
            end;

        end;

    end;
    
else % Fourier pad (upsample/zoom out):

    if( size(inref,3)==1 )
%         fprintf('upsampling...\n');
        inref_F_cut=double(smap.cropOrPad(inref_F,[finalSize(1),finalSize(end)]));
        outref=real(fftshift(ifftn(ifftshift(inref_F_cut))));
        switch method
            case 'fixedSize'
                outref=smap.cropOrPad(smap.cutj(outref,[os(1),os(2)]),[os(1),os(2)],0);
            otherwise
                outref=outref;
        end;
        outref=outref.*(sf.^2);        

    else
        inref_F_cut=double(smap.cropOrPad(inref_F,finalSize));
        outref=real(fftshift(ifftn(ifftshift(inref_F_cut))));        
        outref=outref.*(sf.^3);
        if( strcmp(method,'fixedSize') )
            if( size(outref,1)<os(1) )
                outref=smap.cropOrPad(outref,os,0);
            elseif( size(outref,1)>os(1) )
                outref=smap.cropOrPad(outref,os);
            end;
        end;
    end;
    
end;



if( 1==0 )
%%

s_i=3456;
pp_orig=1.032;
pp_target=1.5;
factor=pp_orig./pp_target;
% factor=1.5./pp_orig;
% pp_target=factor.*pp_orig;

z=(smap.resize_F(ones(s_i),factor,'newSize')); 

s_z=size(z,1)

pp_target
pp_new=(pp_orig * (s_i/s_z))

s_i.*pp_orig
s_z.*pp_new

%%
s_i=3456;
pp_orig=1.5;
pp_target=1.032;
factor=pp_orig./pp_target;
% pp_target=factor.*pp_orig;

z=(smap.resize_F(ones(s_i),factor,'newSize')); 

s_z=size(z,1)

pp_target
pp_new=(pp_orig * (s_i/s_z))

s_i.*pp_orig
s_z.*pp_new

%%
end;

    