function [outref,mask,D]=mask_a_volume(mapref,mask_params,varargin);
% function [outref,mask]=mask_a_volume(mapref,mask_params,varargin);
% 
% returns masked volume and mask
% mode: 'mask' (default): raised cosine map; mask_params: [near_edge cos_width]
% mode: 'shell': shell around the perimeter; mask_params: [d_shell t_shell]
%   d_shell: distance (radius) between edge of map features and center of shell
%   t_shell: half-width (radius) of shell
% mapref is the map
% 
%%
mask_mode='mask';
if( nargin>2 )
    mask_mode=varargin{1};
end;

switch mask_mode
    case 'mask'
        bgVal=mode(mapref(:));
        mm=mapref-bgVal;
        thr=std(abs(mm(:)));
        BW=abs(mm)>thr;
        
        D=bwdist(BW,'euclidean');

        nearEdge=mask_params(1);
        cosWidth=mask_params(2);
        farEdge=nearEdge+cosWidth;

        D11=D;
        D11(D(:)<=nearEdge)=0;
        D11(D(:)>=farEdge)=pi;
        inds_between=find(D(:)>nearEdge & D(:)<farEdge);
        temp=D11(inds_between);
%         temp=temp-nearEdge+1; % mod 052421/jpr
        temp=temp-nearEdge; % mod 052421/jpr
        temp=(temp./cosWidth).*pi;
        temp=(temp);
        D11(inds_between)=temp;
        mask=(cos(D11)./2)+0.5;

    case 'shell'
        
        bgVal=mode(mapref(:));
        mm=mapref-bgVal;
        thr=std(abs(mm(:)));
%         thr=0.5.*std(abs(mm(:)));
        BW=abs(mm)>thr;

        % replace the initial binary mask with one based on a cosine edge
        % mask. This should dilate most of the tight spaces in the interior
        % of the structure where a mask is not sought.
        if( 1 )
            D=bwdist(BW,'euclidean');
            nearEdge=0; 
            cosWidth=1;
            farEdge=nearEdge+cosWidth;
            D11=D;
%             D11(D(:)<nearEdge)=0;
%             D11(D(:)>farEdge)=pi;
%             inds_between=find(D(:)>=nearEdge & D(:)<=farEdge);
            D11(D(:)<=nearEdge)=0;
            D11(D(:)>=farEdge)=pi;
            inds_between=find(D(:)>nearEdge & D(:)<farEdge);
            temp=D11(inds_between);
%             temp=temp-nearEdge+1; % mod 052421/jpr
            temp=temp-nearEdge; % mod 052421/jpr
            temp=(temp./cosWidth).*pi;
            temp=(temp);
            D11(inds_between)=temp;
            mask=(cos(D11)./2)+0.5;
            BW=mask>0;
        end;
        
        D=bwdist(BW,'euclidean');
        d_shell=mask_params(1);
        t_shell=mask_params(2);
        D11=D;
        D11(D11(:)==0)=nan;
        D11=abs(D11-d_shell);
        D11(D11(:)>(t_shell))=nan;
        D11(find(isnan(D11(:))==0))=1;
        D11(find(isnan(D11(:))==1))=0;
        mask=D11;

end;

outref=(mm.*mask)+bgVal;













%%