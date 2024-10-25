function [V_s,V_ctrl_s,qOut,qOut_grid,inds,V_inds]=q_to_density(qA,varargin);
% 
% qA: rotation matrices for which to estimate the density
% varargin: 
%   {1}: rotation matrices to use for the floor (control set)
%   {2}: dilation_degrees - outer range of grid to generate based on
%       significant points (if any) found in the density
% 
% probably the correct way to set a threshold is with a radial max
% 
% function [V_s,V_ctrl_s,qOut,qOut_grid,inds]=q_to_density(qA,varargin);

%%


%constrained_search_flag=0;
constrained_search_flag=1;

% sampling_factor=1;

if( constrained_search_flag )
    N=1024; deg_per_voxel=0.35; % 0.35 deg
%     N=512; deg_per_voxel=0.70; % 0.70 deg
else
    N=512; deg_per_voxel=0.70; % 0.70 deg
end;
V_inds=[];
ctrl_flag=1;
dilation_degrees=2.5;
ind_flag=0;
if( nargin>1 )
    if( size(varargin{1},1)==3 & size(varargin{1},2)==3 )
        qB=varargin{1};
    else
        N=varargin{1}; deg_per_voxel=358.4/N % gives 0.35 deg per voxel @ 1024, 0.7 @ 512
        seedWords={'mt19937ar','Seed',eval('1')};
        stream = RandStream(seedWords{1},seedWords{2},seedWords{3});
        RandStream.setGlobalStream(stream);

        qB=normalizeRM(squeeze(RotationMatrix(quaternion.randRot(1,size(qA,3)))));        
    end;
    if( nargin>2 )
        dilation_degrees=varargin{2}
    end;
    if( nargin>3 )
        ind_flag=1;
    end;
    
else
    seedWords={'mt19937ar','Seed',eval('1')};
    stream = RandStream(seedWords{1},seedWords{2},seedWords{3});
    RandStream.setGlobalStream(stream);
    qB=normalizeRM(squeeze(RotationMatrix(quaternion.randRot(1,size(qA,3)))));
end;

fprintf('converting to rotation vectors...');
pts=zeros(size(qA,3),3);
for i=1:size(qA,3)
    pts(i,:)=rotationMatrixToVector(qA(:,:,i));
end;

if( ctrl_flag )
    pts_ctrl=zeros(size(qB,3),3);
    for i=1:size(qB,3)
        pts_ctrl(i,:)=rotationMatrixToVector(qB(:,:,i));
    end;
    all_pts=cat(1,pts,pts_ctrl);
else
    all_pts=pts;
end;

mm=[min(all_pts(:)) max(all_pts(:))];
fprintf('\n');

cp=floor(N./2)+1;

pts_map=pts.*(-(N-1)./(2.*pi))+((1+N)./2);

tic
V=zeros(N,N,N,'single');
pts_map_r=round(pts_map);
for i=1:size(pts_map_r,1)
    V(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3))=V(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3)) + 1;
end;
if( ind_flag )
    V_inds=cell(N,N,N);
    for i=1:size(pts_map_r,1)
        V_inds{pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3)}(end+1)=i;%V_inds{pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3)}
    end;
end;

r_temp=sqrt(sum((pts_map_r-cp).^2,2));
if( ( length(find(r_temp<(cp-20)))./length(r_temp) )<0.99 )
    disp('skipping crop...');
    constrained_search_flag=0;
end;

VV=[];
if( constrained_search_flag==0 )
    % % full-size:
    rr=smap.rrj(ones(N,N,N)).*N;
    g=single(g2(rr,[1,100])); % sigma_rs=1.45 (x0.35 deg = 0.51 deg)
    factor=size(pts,1)/((4/3)*pi*(cp^3));
    clear rr;
    V_s=smap.iftj(smap.ftj(V).*g);
else
    % % cropped:
    rtu=2*ceil(max(r_temp))+11;
    rtu=128
    VV=smap.cutj(V,rtu.*[1,1,1]);
    rr=smap.rrj(ones(rtu,rtu,rtu)).*rtu;
    g=g2(rr,[1,1.63]); %g=g.*cp;
    g=smap.ftj(g);
    VV_s=smap.iftj(smap.ftj(VV).*g);
    %VV_s=VV_s.*(sum(VV(:))./sum(VV_s(:)));
    V_s=VV_s;    

%     V_s=V_s.*(size(qA,3)./size(qB,3));
end;
% V_s=V_s./deg_per_voxel;

toc
V_ctrl_s=[]; qOut=[]; inds=[]; VV_ctrl=[];
if( ctrl_flag )
    pts_ctrl_map=pts_ctrl.*(-(N-1)./(2.*pi))+((1+N)./2);
    V_ctrl=zeros(N,N,N,'single');
    pts_ctrl_map_r=round(pts_ctrl_map);
    for i=1:size(pts_ctrl_map_r,1)
        V_ctrl(pts_ctrl_map_r(i,1),pts_ctrl_map_r(i,2),pts_ctrl_map_r(i,3))=V_ctrl(pts_ctrl_map_r(i,1),pts_ctrl_map_r(i,2),pts_ctrl_map_r(i,3))+1;
    end;
    
    if( constrained_search_flag==0 )
        V_ctrl_s=smap.iftj(smap.ftj(V_ctrl).*g);
    else
        VV_ctrl=smap.cutj(V_ctrl,rtu.*[1,1,1]);
        VV_ctrl_s=smap.iftj(smap.ftj(VV_ctrl).*g);
        %VV_ctrl_s=VV_ctrl_s.*(sum(VV_ctrl(:))./sum(VV_ctrl_s(:)));
        V_ctrl_s=VV_ctrl_s;
    end;
else
    return
end;



if( constrained_search_flag )
    thr=max(V_ctrl_s(:))
    
%     disp('using difference and zero threshold');
%     thr=0
% %     V_ctrl_s=(V_ctrl_s.*(sum(V_s(:))./sum(V_ctrl_s(:))));
%     V_s=V_s-V_ctrl_s;
% %     V_s=V_s+min(V_s(:));
else
    thr=1.5.*max(V_ctrl_s(:))
%     thr=2.*max(V_ctrl_s(:)) % 020520, testing
%     thr=1.2.*max(V_ctrl_s(:)) % 030520, testing (not used after all)
end;

if( size(V_s,1) < N )
    V_s=smap.extendj(V_s,N.*[1,1,1],0);
    V_ctrl_s=smap.extendj(V_ctrl_s,N.*[1,1,1],0);
end;

% % for SH:
if( 0 )
    smap.plotSH(V_ctrl_s(:),'r');
    smap.plotSH(V_s(:),'b');
    set(gca,'xscale','log');
    xlim([2e-3 2]), ylim([5e-1 5e7]);
    grid off, box off
    xlabel('rotation density (pts/deg^3)');
    ylabel('survival count');
    smap.avl(thr);
    export_fig('~/OneDrive/figs/LSU_body_SH_density_init.eps');
end;

inds_thr=find(V_s(:)>(thr));
V_thr=V_s>thr;
% V_ctrl_thr=V_ctrl_s>thr;
inds=[];
% inds_ctrl=[];
if( ~isempty(inds_thr) )    
    inds=[];
    for i=1:size(pts_map_r,1)
        if( V_thr(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3))>0 )
            inds=[inds i];
        end;
    end;
    
    % convert back to rotation vectors, then matrices:
    rv=pts(inds,:); % the original points
    qOut=zeros(3,3,size(rv,1));
    for i=1:size(rv,1)
        qOut(:,:,i)=rotationVectorToMatrix(rv(i,:));
    end;
    qOut=normalizeRM(qOut);

%     inds_ctrl=[];
%     for i=1:size(pts_ctrl_map_r,1)
%         if( V_ctrl_thr(pts_ctrl_map_r(i,1),pts_ctrl_map_r(i,2),pts_ctrl_map_r(i,3))>0 )
%             inds_ctrl=[inds_ctrl i];
%         end;
%     end;
%     rv_ctrl=pts_ctrl(inds_ctrl,:); % the original points
%     qOut_ctrl=zeros(3,3,size(rv_ctrl,1));
%     for i=1:size(rv_ctrl,1)
%         qOut_ctrl(:,:,i)=rotationVectorToMatrix(rv_ctrl(i,:));
%     end;
%     qOut_ctrl=normalizeRM(qOut_ctrl);

    if( dilation_degrees>0 )
        V_thr=smap.cutj(V_thr,256.*[1,1,1]);
        [outref,mask]=mask_a_volume(V_thr,[0 (1./deg_per_voxel).*(dilation_degrees.*2)],'mask');
        mask=single(mask>0.5);
        V_thr=smap.extendj(mask,N.*[1,1,1],0);

        % % formerly outside the loop:
        inds_thr=find(V_thr(:)==1);
        [nx,ny,nz]=ind2sub([N N N],inds_thr);
        
        rv_grid=([nx ny nz]-((1+N)./2))./(-(N-1)./(2.*pi)); % gridded points
        qOut_grid=zeros(3,3,size(rv_grid,1));
        for i=1:size(rv_grid,1)
            qOut_grid(:,:,i)=rotationVectorToMatrix(rv_grid(i,:));
        end;
        qOut_grid=normalizeRM(qOut_grid);
        % % :outside
    else
        qOut_grid=[];
    end;
    
else
    qOut=[];
    qOut_grid=[];
%     inds=[];
    fprintf('no significant points found\n');
end;

toc























%% v1.0: before cleaning up

% 
% crop_flag=1;
% 
% N=512; deg_per_voxel=0.70; % 0.70 deg
% % N=1024; deg_per_voxel=0.35; % 0.35 deg
% 
% 
% ctrl_flag=0;
% if( nargin>1 )
%     qB=varargin{1};
%     ctrl_flag=1;
% end;
% 
% fprintf('converting to rotation vectors...');
% pts=zeros(size(qA,3),3);
% for i=1:size(qA,3)
%     pts(i,:)=rotationMatrixToVector(qA(:,:,i));
% end;
% 
% if( ctrl_flag )
% pts_ctrl=zeros(size(qB,3),3);
% for i=1:size(qB,3)
%     pts_ctrl(i,:)=rotationMatrixToVector(qB(:,:,i));
% end;
% all_pts=cat(1,pts,pts_ctrl);
% else
%     all_pts=pts;
% end;
% 
% mm=[min(all_pts(:)) max(all_pts(:))];
% fprintf('\n');
% 
% cp=floor(N./2)+1;
% 
% pts_map=pts.*(-(N-1)./(2.*pi))+((1+N)./2);
% 
% tic
% V=zeros(N,N,N,'single');
% pts_map_r=round(pts_map);
% for i=1:size(pts_map_r,1)
%     V(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3))=V(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3)) + 1;
% end;
% 
% r_temp=sqrt(sum((pts_map_r-cp).^2,2));
% if( max(r_temp)>(cp-10) )
%     disp('skipping crop...');
%     crop_flag=0;
% end;
% 
% VV=[];
% if( crop_flag==0 )
%     % % full-size:
%     rr=smap.rrj(ones(N,N,N)).*N;
%     g=single(g2(rr,[1,100])); % sigma_rs=1.45 (x0.35 deg = 0.51 deg)
%     factor=size(pts,1)/((4/3)*pi*(cp^3));
% %     the_sinc=1.67.*factor.*sinc(rr./N).^2;
%     clear rr;
% %     V=V-the_sinc;
%     V_s=smap.iftj(smap.ftj(V).*g);
% else
%     % % cropped:
%     rtu=2*ceil(max(r_temp))+11;
%     rtu=128
%     VV=smap.cutj(V,rtu.*[1,1,1]);
%     rr=smap.rrj(ones(rtu,rtu,rtu)).*rtu;
%     g=g2(rr,[1,1.63]); %g=g.*cp;
%     g=smap.ftj(g);
%     VV_s=smap.iftj(smap.ftj(VV).*g);
%     VV_s=VV_s.*(sum(VV(:))./sum(VV_s(:)));
%     V_s=VV_s;    
% end;
% 
% toc
% V_ctrl_s=[]; qOut=[]; inds=[]; VV_ctrl=[];
% if( ctrl_flag )
%     pts_ctrl_map=pts_ctrl.*(-(N-1)./(2.*pi))+((1+N)./2);
%     V_ctrl=zeros(N,N,N,'single');
%     pts_ctrl_map_r=round(pts_ctrl_map);
%     for i=1:size(pts_ctrl_map_r,1)
%         V_ctrl(pts_ctrl_map_r(i,1),pts_ctrl_map_r(i,2),pts_ctrl_map_r(i,3))=V_ctrl(pts_ctrl_map_r(i,1),pts_ctrl_map_r(i,2),pts_ctrl_map_r(i,3))+1;
%     end;
% 
%     if( crop_flag==0 )
% %         V_ctrl=V_ctrl-the_sinc;
%         V_ctrl_s=smap.iftj(smap.ftj(V_ctrl).*g);
%     else
%         VV_ctrl=smap.cutj(V_ctrl,rtu.*[1,1,1]);
%         VV_ctrl_s=smap.iftj(smap.ftj(VV_ctrl).*g);
%         VV_ctrl_s=VV_ctrl_s.*(sum(VV_ctrl(:))./sum(VV_ctrl_s(:)));
%         V_ctrl_s=VV_ctrl_s;
%     end;
%     
% else
%     V_ctrl_s=V_s./(10.*std(V_s(:)));
% end;
% 
% V_s=V_s./deg_per_voxel;
% V_ctrl_s=V_ctrl_s./deg_per_voxel;
% 
% 
% 
% V_ctrl_s=V_ctrl_s.*(sum(V_s(:))./sum(V_ctrl_s(:)));
% 
% 
% 
% if( length(find(VV_ctrl(:)>0)) > length(find(VV(:)>0)) )
%     V_s=V_s-V_ctrl_s;
%     thr=0.5
% else
%     thr=1.5.*max(V_ctrl_s(:))
% end;
% 
% if( size(V_s,1) < N )
%     V_s=smap.extendj(V_s,N.*[1,1,1],0);
%     V_ctrl_s=smap.extendj(V_ctrl_s,N.*[1,1,1],0);
% end;
% 
% inds_thr=find(V_s(:)>(thr));
% V_thr=V_s>thr;
% inds=[];
% % [nx,ny,nz]=ind2sub([N N N],inds_thr);
% if( ~isempty(inds_thr) )
% 
% 
% %     % % also return the indices of input RMs that ended up above the threshold:
% %     [li_x,lib]=ismember(pts_map_r(:,1),nx);
% %     inds_x=find(li_x==1);
% %     [li_y,lib]=ismember(pts_map_r(:,2),ny);
% %     inds_y=find(li_y==1);
% %     [li_z,lib]=ismember(pts_map_r(:,3),nz);
% %     inds_z=find(li_z==1);
% %     inds=intersect(intersect(inds_x,inds_y),inds_z);
% %     
% %     inds=[];
% %     for j=1:length(nx)
% %         inds=[inds; find(pts_map_r(:,)==[nx(j) ny(j) nz(j)])];
% %     end;
%     
%     inds=[];
%     for i=1:size(pts_map_r,1)
%         %     cts(i)=cts(i)+1;
%         %     V(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3))=inds(i);%
%         if( V_thr(pts_map_r(i,1),pts_map_r(i,2),pts_map_r(i,3))>0 )
%             inds=[inds i];
%         end;
%     end;
%     
%     % convert back to rotation vectors, then matrices:
%     rv=pts(inds,:); % the original points
%     qOut=zeros(3,3,size(rv,1));
%     for i=1:size(rv,1)
%         qOut(:,:,i)=rotationVectorToMatrix(rv(i,:));
%     end;
%     qOut=normalizeRM(qOut);
%     
%     dilation_degrees=2.5;
%     V_thr=smap.cutj(V_thr,256.*[1,1,1]);
%     [outref,mask]=mask_a_volume(V_thr,[0 (1./deg_per_voxel).*(dilation_degrees.*2)],'mask');
%     mask=single(mask>0.5);
%     V_thr=smap.extendj(mask,N.*[1,1,1],0);
%     inds_thr=find(V_thr(:)==1);
%     [nx,ny,nz]=ind2sub([N N N],inds_thr);
%     
%     rv_grid=([nx ny nz]-((1+N)./2))./(-(N-1)./(2.*pi)); % gridded points
%     qOut_grid=zeros(3,3,size(rv_grid,1));
%     for i=1:size(rv_grid,1)
%         qOut_grid(:,:,i)=rotationVectorToMatrix(rv_grid(i,:));
%     end;
%     qOut_grid=normalizeRM(qOut_grid);
%     
% else
%     qOut=[];
%     qOut_grid=[];
% %     inds=[];
%     fprintf('no significant points found\n');
% end;
% 
% toc
 




%%
% test=((pts_map_r-1)./(N-1)).*sum(abs(mm))+mm(1);

% dsf=2;
% 
% V_s=smap.resize_F(V_s,1/dsf,'newSize');
% V_ctrl_s=smap.resize_F(V_ctrl_s,1/dsf,'newSize');

% V_diff=(V_s-V_ctrl_s);


% clear the_sinc;
% ts=the_sinc.*(1./max(the_sinc(:)));
% thr=1;

% %inds=find(rr(:)<cp);
% thr=sqrt(2).*erfcinv(2./(nSamples));
% % thr=nSamples
% for i=1:cp
%     inds=find(rr(:)>=(i-1) & rr(:)<(i));
%     vs(i)=sum(V(inds));
%     li(i)=length(inds);
% end;

% inds_lin=sub2ind([N N N],pts_map_r(:,1),pts_map_r(:,2),pts_map_r(:,3));
% nSamples=size(pts_map,1);
% ui=unique(inds);
% cts=zeros(1,length(ui));
% ii=randi([1 length(inds)],1,length(pts));
% inds=inds(ii);
% ii=randperm(length(inds));
%inds=inds(ii);
% Vi=zeros(N,N,N,'single');
% test=(rr.^3-(rr-1).^3);


%%

