function outref=templates(obj,qref,dfs,varargin);
% function templates=smap_templates(qref,dfs,edgeSize,pixelSize,SPVname,varargin);
% templates=smap_templates(qref,dfs,edgeSize,pixelSize,SPVname,varargin);
% dfs are [df1 df2 ast] following the CTFFind convention (but df1 and df2 are in nm, not Angstroms)
% edgeSize in pixels
% pixelSize is in Angstroms
% SPVname refers to a directory name in ~/matching/maps/ with an SPV.img file
% varargin is up to 2 variables, which are
% 1) envFlag (0 or 1; 0 means no coherence envelope; default=1)
% 2) MTF parameters ([a b c alpha beta]; default=[0 0.935 0 0 0.64])
% a perfect MTF is [0.5 0.5 0 0 0]
%
% tip: for a large # of templates all at one defocus, give one [df df ast]
% to avoid calculating N different CTFs
%
%

%pause;

% % compile with: mcc -m -v -R -nojvm -R -singleCompThread smap_templates.m
if( isempty(dfs)==1 )
    method='exitWave';
else
    method='detectorIntensity';
    ppmFlag=0;
    if( ~isempty(strfind(dfs(1,:),'PPM')) )
        ppmFlag=1;
    end;
end;

if( strcmp(class(qref),'quaternion') )
    qref=squeeze(RotationMatrix(qref));
end;

%F_abs=0.075;
F_abs=0;

switch method
    case 'detectorIntensity'
        
%        nTemplates=size(qref,3);
        nR=size(qref,3);
        nDf=size(dfs,1);
        nTemplates=max([nR nDf]);
        if( nR==1 )
            temp=zeros(3,3,nDf);
            for i=1:nDf
                temp(:,:,i)=qref(:,:,1);
            end;
            qref=temp;
        end;
                
        disp(['computing ' num2str(max([nTemplates,size(dfs,1)])) ' templates (' datestr(now) ')']);
        
        baseDir='~/matching/';
        
        % flips CTFFind astigmatism:
        if( size(dfs,2)==3 )
            %dfs(:,3)=-dfs(:,3); % 052417
        end;
        params.df=dfs;
        params.envFlag=1;
        params.MTF=[0 0.935 0 0 0.64];
        params.aPerPix=obj.prop.nmPerPixel_SP*10;
        nDfs=size(params.df,1);
        if( nDfs>nTemplates )
            fprintf('computing templates for one orientation at %d defocus values...\n',nDfs);
            for j=1:nDfs
                qref(:,:,j)=qref(:,:,1);
            end;
        end;
        
        % read in the scattering potential and forward transform (leave origin at center):
        fprintf(['reading in scattering potential for ' obj.ID.ID '...\n']);
%         SPV=smap.ri(smap.checkBaseDir(obj.prop.SPName));
        SPV=smap.ri(obj.prop.SPName);
        Npix=size(SPV,1);
        
        
%        pause
        
        
        [k_2d,centerPixel]=smap.getKs(zeros(Npix,Npix),params.aPerPix);
        [x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
        cp=floor(Npix./2)+1;
        x0=x(:,:,cp)-cp;
        y0=y(:,:,cp)-cp;
        z0=zeros(Npix,Npix); 
        
%         % % ewald correction:
%         k_ny=k_2d(cp,end);
%         x0_temp=x0.*k_ny./cp;%max(abs(x0(cp,:)));
%         y0_temp=y0.*k_ny./cp;%max(abs(y0(:,cp)));
%         temp=smap.def_consts();        
%         V=temp.V; %300e3;
%         %V=1e6;
%         m_e = 9.10938215e-31; % kg
%         lambda = temp.h/sqrt(temp.q_e*V*m_e*(temp.q_e/m_e*V/temp.c^2 + 2 ));
%         lambda=lambda.*1e10;        
%         Rmax=(1./lambda).*params.aPerPix;
%         z0=real(Rmax-sqrt((Rmax.^2)-(x0_temp.^2+y0_temp.^2)));
%         z0=z0.*cp./k_ny;
%         max(z0(:));

        xyz=[x0(:) y0(:) z0(:)];
        clear x y z;
        
%         V=SPV;
        V=SPV+1i*F_abs*SPV;
        V_F=fftshift(fftn(ifftshift(V)));
        V_F(cp,cp,cp)=0;
        clear SPV V;

        xyz_test=xyz;%(61251:end,:);
        dummyX_test=[1:size(V_F,1)]; dummyY_test=[1:size(V_F,2)]; dummyZ_test=[0:(cp-1)]-(floor(cp./2)+1);%size(V_F,3);
        
        % compute the CTF(s) and MTF:
        fprintf('computing CTF(s) and MTF...\n');
        
        if( nargin>3 )
            params.envFlag=varargin{1};
            if( params.envFlag==0 )
                disp('no coherence envelope...');
            end;
            if( nargin>4 )
                params.MTF=varargin{2};
                fprintf(['using MTF with parameters:\n' num2str(params.MTF(1)) '\n' num2str(params.MTF(2)) '\n' num2str(params.MTF(3)) '\n' num2str(params.MTF(4)) '\n' num2str(params.MTF(5)) '\n']);
            end;
        end;
        
        %MTF=smap.approxMTF(k_2d,params.MTF,1);
        MTF=ones(size(k_2d,1),size(k_2d,2));

        CTF=zeros(Npix,Npix,nDfs);
        
        if( ppmFlag==1 )
            MTFCTF=smap.makePhasePlate(zeros(Npix,Npix,'single'),'vulovic');
        else
            
            if( nDfs>1 )
                if( nDfs<=1e9 ) %300 )
                    for i=1:nDfs
                        %             [sfs_CTF,~,CTF(:,:,i)]=makeCTF(params.df(i,:),Npix,params.aPerPix,params.envFlag,0,0);
                        CTF(:,:,i)=smap.ctf(params.df(i,:),Npix,params.aPerPix,params.envFlag,0,0);
                        MTFCTF(:,:,i)=MTF.*CTF(:,:,i);
                    end;
                else
                    disp('using mean defocus for large set...');
                    CTF(:,:,i)=smap.ctf(params.df(i,:),Npix,params.aPerPix,params.envFlag,0,0);
                    %         [sfs_CTF,~,CTF]=makeCTF(mean(params.df),Npix,params.aPerPix,params.envFlag,0,0);
                    for i=1:nDfs
                        MTFCTF(:,:,i)=MTF.*CTF;
                    end;
                end;
            else
                CTF=smap.ctf(params.df,Npix,params.aPerPix,params.envFlag,0,0);
                %     [sfs_CTF,~,CTF]=makeCTF(params.df,Npix,params.aPerPix,params.envFlag,0,0);
                MTFCTF=MTF.*CTF;
            end;
        end;
        
        R_opt=[1     1    -1
            1     1    -1
            -1     -1     1];
        
        templates=zeros(Npix,Npix,nTemplates);
        for i=1:nTemplates            
            RM=qref(:,:,i)';%.*R_opt; % this preserves the old quaternion flip. R was calculated directly from hopf set
            xyz_r=(RM*xyz')';
            X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
            
            output_image = complex(ba_interp3(double(real(V_F)),X,Y,Z,'cubic'),ba_interp3(double(imag(V_F)),X,Y,Z,'cubic'));
            vi_rs=reshape(output_image,Npix,Npix);
            vi_rs(find(isnan(vi_rs)==1))=0;
            projPot=vi_rs;            
            %projPot=vi_rs+1i*F_abs*vi_rs;
            
            % multiply by the aberration function, compute detector intensity, apply MTF:
            if( nDfs>1 )
                %         template=1+2.*(fftshift(real(ifftn(ifftshift(MTFCTF(:,:,i).*projPot)))));
                %         template=1+2.*(fftshift(real(ifftn(ifftshift(-imag(MTFCTF(:,:,i)).*projPot)))));
%                ew=exp(1i.*(fftshift(real(ifftn(ifftshift(projPot))))));
                ew=exp(1i.*(fftshift((ifftn(ifftshift(projPot))))));
                w_det=fftshift(ifftn(ifftshift(fftshift(fftn(ifftshift(ew))).*MTFCTF(:,:,i))));
                template=w_det.*conj(w_det);
            else
                % % % weak-phase approximation:
                %         template=1+2.*(fftshift(real(ifftn(ifftshift(MTFCTF.*projPot)))));
                %         template=1+2.*(fftshift(real(ifftn(ifftshift(-imag(MTFCTF).*projPot)))));
                %         ew=exp(1i.*(fftshift((ifftn(ifftshift(projPot))))));
                % % thick:
%                ew=exp(1i.*(fftshift(real(ifftn(ifftshift(projPot))))));
                ew=exp(1i.*(fftshift((ifftn(ifftshift(projPot))))));
                w_det=fftshift(ifftn(ifftshift(fftshift(fftn(ifftshift(ew))).*MTFCTF)));
                template=w_det.*conj(w_det);
            end;
            templates(:,:,i)=template;
            if( mod(i,10)==0 )
                fprintf('%d/%d\n',i,nTemplates);%length(qref));
            end;
        end;
        
        outref=single(templates);
        
        disp(['done (' datestr(now) ')']);
        
    case 'exitWave'
        
        nTemplates=size(qref,3);
        
        disp(['computing ' num2str(max([nTemplates,size(dfs,1)])) ' templates (' datestr(now) ')']);
        
        params.envFlag=1;
        params.MTF=[0 0.935 0 0 0.64];
        params.aPerPix=obj.prop.nmPerPixel_SP*10;
        
        % read in the scattering potential and forward transform (leave origin at center):
        fprintf(['reading in scattering potential for ' obj.ID.ID '...\n']);
        SPV=smap.ri(smap.checkBaseDir(obj.prop.SPName));
        Npix=size(SPV,1);
        
        [k_2d,centerPixel]=smap.getKs(zeros(Npix,Npix),params.aPerPix);
        [x,y,z]=meshgrid(1:Npix,1:Npix,1:Npix);
        cp=floor(Npix./2)+1;
        x0=x(:,:,cp)-cp;
        y0=y(:,:,cp)-cp;
        z0=zeros(Npix,Npix);
                
%         % % ewald correction
%         k_ny=k_2d(cp,end);
%         x0_temp=x0.*k_ny./cp;%max(abs(x0(cp,:)));
%         y0_temp=y0.*k_ny./cp;%max(abs(y0(:,cp)));
%         temp=smap.def_consts();        
%         V=cc.V; %300e3;
%         %V=1e6;
%         m_e = 9.10938215e-31; % kg
%         lambda = temp.h/sqrt(temp.q_e*V*m_e*(temp.q_e/m_e*V/temp.c^2 + 2 ))
%         lambda=lambda.*1e10;
%         %lambda=temp.wl.*1e10; % in A
%         
%         Rmax=(1./lambda).*params.aPerPix
%         z0=real(Rmax-sqrt((Rmax.^2)-(x0_temp.^2+y0_temp.^2)));
%         z0=z0.*cp./k_ny;
%         max(z0(:))
        
        xyz=[x0(:) y0(:) z0(:)];
        clear x y z;
        
%         V=SPV;
        V=SPV+1i*F_abs*SPV;
        V_F=fftshift(fftn(ifftshift(V)));
        V_F(cp,cp,cp)=0;
        
        clear SPV V;
        
        R_opt=[1     1    -1
            1     1    -1
            -1     -1     1];
        
        templates=zeros(Npix,Npix,nTemplates);
        for i=1:nTemplates
            
            RM=qref(:,:,i)';%.*R_opt; % this preserves the old quaternion flip. R was calculated directly from hopf set
            xyz_r=(RM*xyz')';
            X=xyz_r(:,1)+cp; Y=xyz_r(:,2)+cp; Z=xyz_r(:,3)+cp;
            
            output_image = complex(ba_interp3(double(real(V_F)),X,Y,Z,'cubic'),ba_interp3(double(imag(V_F)),X,Y,Z,'cubic'));
            vi_rs=reshape(output_image,Npix,Npix);
            vi_rs(find(isnan(vi_rs)==1))=0;
            %projPot=vi_rs;
            projPot=vi_rs+1i*F_abs*vi_rs;

            templates(:,:,i)=projPot;
            
            if( mod(i,10)==0 )
                fprintf('%d/%d\n',i,nTemplates);%length(qref));
            end;
        end;
        
        outref=single(templates);
        
        disp(['done (' datestr(now) ')']);
        
end;


%%













