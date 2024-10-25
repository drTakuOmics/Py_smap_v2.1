function outref=rotate3dMatrix(inref,R,varargin);

%method='cpu';
%method='gpu';

flag=0;
if( nargin>2 )
    flag=varargin{1};
end;

if( isa(inref,'gpuArray') )
    method='gpu';
else
    method='cpu';
end;

Npix=size(inref,1);
Npix_fc=size(inref,3);
cp=gather(floor(Npix./2)+1);

if( isreal(inref) )
    inputType='realspace';
%     outref=inref;
    rr_for_sinc=smap.rrj(ones(Npix,Npix,Npix,'single'));
    rs_sinc=sinc(double(rr_for_sinc)).^2;
    clear rr_for_sinc;
    inref=inref./rs_sinc;
    clear rs_sinc;
    inref=smap.ftj(inref);
else
    inputType='kspace';
end;



        dcVal=inref(cp,cp,cp);
        inref(cp,cp,cp)=0;
        V_Fr=real(inref); V_Fi=imag(inref);
        %clear inref;
        vec=([-((cp-1)):((cp-1))]);
        if( mod(Npix,2)==0 )
            vec=vec(1:end-1);
        end;

        [xd,yd,zd]=meshgrid(vec,vec,vec);
        xd=xd(:,:,1:Npix_fc); yd=yd(:,:,1:Npix_fc); zd=zd(:,:,1:Npix_fc);
        xyz=([xd(:) yd(:) zd(:)]);
        clear xd yd zd;
        
%         tic;
        
        if( flag ) %Npix_fc < Npix )
            RR=(sqrt(sum(xyz.^2,2)));
            inds=find((RR<(cp-1)) & xyz(:,3)>=0);
            xyz=xyz(inds,:);        
            xyz_r=(xyz*R)+cp; % must be double-precision

            %inds=find((RR<(cp-1))&(xyz_r(:,3)>=0));
            amp_factor=2;
        else
            xyz_r=(xyz*R); % must be double-precision
            RR=(sqrt(sum(xyz_r.^2,2)));
            inds=find(RR<(cp-1));
            xyz_r=xyz_r(inds,:)+cp;
            amp_factor=1;
        end;
                
        clear RR;
        
        switch method
            case 'cpu'
                %output_vol=zeros(Npix,Npix,Npix_fc,'single');
                X=xyz_r(:,1);
                Y=xyz_r(:,2);
                Z=xyz_r(:,3);
                clear xyz_r;
                
                dummy=[1:Npix];                
                V_Fr=real(inref);
                temp=interpn(dummy,dummy,dummy,V_Fr,Y,X,Z); % 19 s
                V_Fr=V_Fr.*0;
                V_Fr(inds)=temp;
                V_Fi=imag(inref);
                temp=interpn(dummy,dummy,dummy,V_Fi,Y,X,Z); % 19 s
                V_Fi=V_Fi.*0;
                V_Fi(inds)=temp;
                outref=complex(V_Fr,V_Fi);
                outref(cp,cp,cp)=dcVal;
%                 temp=interpn(dummy,dummy,dummy,inref,Y,X,Z,'linear');
%                 outref(inds)=temp;

            case 'gpu'
                %output_vol=gpuArray.zeros(Npix,Npix,Npix_fc,'single');
                %clear xyz;
                x1=[1:Npix];
                x2=x1;
                x3=x1;
                Nx1 = length(x1);
                Nx2 = length(x2);
                Nx3 = length(x3);
                x11 = x1(1);
                x21 = x2(1);
                x31 = x3(1);
                s1 = x1(2)-x11;
                s2 = x2(2)-x21;
                s3 = x3(2)-x31;
                
                % cannot do this in 2013a; values above 2^24 get digitized
                %xyz_r=single(xyz_r);
                %inds=single(inds);
                
                X=gpuArray(xyz_r(:,1));
                Y=gpuArray(xyz_r(:,2));
                Z=gpuArray(xyz_r(:,3));                
                %pause;
                
                %output_image=gpuArray.zeros(1,Npix*Npix,'single'); wait(gdev);
                V=V_Fr; V_Fr=V_Fr.*0;
                V_Fr(inds) = arrayfun(@gpuInterp,Y,X,Z);
                %V_Fr(inds)=output_vol_plus;

                V=V_Fi; V_Fi=V_Fi.*0;
                V_Fi(inds) = arrayfun(@gpuInterp,Y,X,Z);
                %V_Fi(inds)=output_vol_plus;

%                 V=V_Fr;
%                 temp = arrayfun(@gpuInterp,Y,X,Z);
%                 V_Fr(inds)=temp;
%                 V=V_Fi;
%                 temp = arrayfun(@gpuInterp,Y,X,Z);
%                 V_Fi(inds)=temp;
                clear V X Y Z inref;
                
                outref=amp_factor.*complex(V_Fr,V_Fi);
                %pause;
                outref(cp,cp,cp)=dcVal;
%                     toc
    
        end;
%     case 'realspace'
%         %
%         % for real-space input
%         %
%         
%         V=inref;
%         clear inref;
%         vec=([-((cp-1)):((cp-1))]);
%         if( mod(Npix,2)==0 )
%             vec=vec(1:end-1);
%         end;
%         [xd,yd,zd]=meshgrid(vec,vec,vec);
%         xyz=([xd(:) yd(:) zd(:)]);        
%         RR=(sqrt(sum(xyz.^2,2)));
%         inds=(find(RR<(cp-1)));
%         xyz=xyz(inds,:);
%         clear xd yd zd;
%         xyz_r=(xyz*R)+cp;
%         clear RR;
%         
%         switch method
%             case 'cpu'
%                 X=xyz_r(:,1);
%                 Y=xyz_r(:,2);
%                 Z=xyz_r(:,3);
%                 clear xyz_r;
%                 
%                 dummy=[1:Npix];
%                 temp=interpn(dummy,dummy,dummy,V,Y,X,Z); % 19 s
%                 outref=V.*0;
%                 outref(inds)=temp;
% %                toc
%                 
%             case 'gpu'
%                 clear xyz;
%                 x1=[1:Npix];
%                 x2=x1;
%                 x3=x1;
%                 Nx1 = length(x1);
%                 Nx2 = length(x2);
%                 Nx3 = length(x3);
%                 x11 = x1(1);
%                 x21 = x2(1);
%                 x31 = x3(1);
%                 s1 = x1(2)-x11;
%                 s2 = x2(2)-x21;
%                 s3 = x3(2)-x31;
%                 
%                 X=gpuArray(xyz_r(:,1));
%                 Y=gpuArray(xyz_r(:,2));
%                 Z=gpuArray(xyz_r(:,3));
%                 clear xyz_r;
%                 
%                 temp = arrayfun(@gpuInterp,Y,X,Z);
%                 outref=V.*0;
%                 outref(inds)=temp;
%                 
%         end;
% 
% end;

switch inputType
    case 'realspace'
        outref=smap.iftj(outref);
    case 'kspace'
        
end;

    function outref=gpuInterp(x1i,x2i,x3i)
                
        x1i = (x1i-x11)/s1+1;
        x2i = (x2i-x21)/s2+1;
        x3i = (x3i-x31)/s3+1;
        
        loc1 = min(Nx1-1,floor(x1i));
        loc2 = min(Nx2-1,floor(x2i));
        loc3 = min(Nx3-1,floor(x3i));
        
        % x1i now becomes weight for x1i. Overwrite instead of creating new
        % variable for memory performance. Same for x2i and x3i.
        x1i = x1i-loc1;
        x2i = x2i-loc2;
        x3i = x3i-loc3;

        outref  = (1-x1i)*((1-x2i)*((1-x3i)*V(loc1  +(loc2-1)*Nx1+(loc3-1)*Nx1*Nx2)   ...
            +    x3i *V(loc1  +(loc2-1)*Nx1+(loc3)  *Nx1*Nx2))  ...
            +    x2i *((1-x3i)*V(loc1  +(loc2)  *Nx1+(loc3-1)*Nx1*Nx2)   ...
            +    x3i *V(loc1  +(loc2)  *Nx1+(loc3)  *Nx1*Nx2))) ...
            +    x1i *((1-x2i)*((1-x3i)*V(loc1+1+(loc2-1)*Nx1+(loc3-1)*Nx1*Nx2)   ...
            +    x3i *V(loc1+1+(loc2-1)*Nx1+(loc3)  *Nx1*Nx2))  ...
            +    x2i *((1-x3i)*V(loc1+1+(loc2)  *Nx1+(loc3-1)*Nx1*Nx2)   ...
            +    x3i *V(loc1+1+(loc2)  *Nx1+(loc3)  *Nx1*Nx2)));
                
    end
%     function outref=gpuInterp(x1i,x2i,x3i)
% %         x1i = (x1i-1)/ds+1;
% %         x2i = (x2i-1)/ds+1;
% %         x3i = (x3i-1)/ds+1;        
%         loc1 = min(Npix-1,floor(x1i));
%         loc2 = min(Npix-1,floor(x2i));
%         loc3 = min(Npix-1,floor(x3i));
%         
%         % x1i now becomes weight for x1i. Overwrite instead of creating new
%         % variable for memory performance. Same for x2i and x3i.
%         x1i = x1i-loc1;
%         x2i = x2i-loc2;
%         x3i = x3i-loc3;
% 
%         outref  = (1-x1i)*((1-x2i)*((1-x3i)*V(loc1  +(loc2-1)*Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1  +(loc2-1)*Npix+(loc3)  *Npix*Npix))  ...
%             +    x2i *((1-x3i)*V(loc1  +(loc2)  *Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1  +(loc2)  *Npix+(loc3)  *Npix*Npix))) ...
%             +    x1i *((1-x2i)*((1-x3i)*V(loc1+1+(loc2-1)*Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1+1+(loc2-1)*Npix+(loc3)  *Npix*Npix))  ...
%             +    x2i *((1-x3i)*V(loc1+1+(loc2)  *Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1+1+(loc2)  *Npix+(loc3)  *Npix*Npix)));
%         
%     end
        

end


%%
% 
% function outref=rotate3dMatrix(inref,R,varargin);
% 
% %method='cpu';
% method='gpu';
% 
% if( nargin>2 )
%     method=varargin{1};
% end;    
% if( strcmp(method,'gpu') )
%     %gdev=gpuDevice(1);
%     %inref=gpuArray(inref);
%     %wait(gdev);
% end;
% 
% Npix=size(inref,1);
% 
% %cp=single(gather(floor(Npix./2)+1)); % leave as double precision !
% cp=gather(floor(Npix./2)+1); % leave as double precision !
% 
% temp=smap.ftj(inref);
% dcVal=temp(cp,cp,cp);
% temp(cp,cp,cp)=0;
% V_Fr=real(temp); V_Fi=imag(temp);
% clear temp;
% 
% %vec=single([-((cp-1)):((cp-1))]);
% vec=double([-((cp-1)):((cp-1))]);
% 
% [xd,yd,zd]=meshgrid(vec,vec,vec);
% 
% %xyz=single([xd(:) yd(:) zd(:)]);
% xyz=double([xd(:) yd(:) zd(:)]);
% 
% %[min(xyz(:)) max(xyz(:)) mean(xyz(:))];
% clear xd yd zd;
% 
% %xyz_r=single(xyz*R);
% xyz_r=double(xyz*R); % must be double-precision
% 
% %RR=single(sqrt(sum((xyz_r).^2,2)));
% RR=double(sqrt(sum(xyz_r.^2,2)));
% 
% inds=find(RR<(cp-1));
% xyz_r=xyz_r(inds,:)+cp;
% %[min(xyz_r(:)) max(xyz_r(:)) mean(xyz_r(:))];
% clear RR;
% 
% switch method
%     case 'cpu'        
%         X=xyz_r(:,1);
%         Y=xyz_r(:,2);
%         Z=xyz_r(:,3);
%         clear xyz_r;
% 
%         dummy=[1:Npix];
%         temp=interpn(dummy,dummy,dummy,V_Fr,Y,X,Z); % 19 s
%         V_Fr(inds)=temp;
%         temp=interpn(dummy,dummy,dummy,V_Fi,Y,X,Z); % 19 s
%         V_Fi(inds)=temp;
%         outref=smap.iftj(complex(V_Fr,V_Fi));
%         toc; tic
%         toc
% 
%         
%     case 'gpu'
%         clear xyz;
%         x1=[1:Npix];
%         x2=x1;
%         x3=x1;
%         Nx1 = length(x1);
%         Nx2 = length(x2);
%         Nx3 = length(x3);
%         x11 = x1(1);
%         x21 = x2(1);
%         x31 = x3(1);
%         s1 = x1(2)-x11;
%         s2 = x2(2)-x21;
%         s3 = x3(2)-x31;
% 
%         X=gpuArray(xyz_r(:,1));
%         Y=gpuArray(xyz_r(:,2));
%         Z=gpuArray(xyz_r(:,3));
%         
%         tic;
%         V=V_Fr;
%         V_Fr(inds) = arrayfun(@gpuInterp,Y,X,Z);
%         V=V_Fi;
%         V_Fi(inds) = arrayfun(@gpuInterp,Y,X,Z);
%         toc
%         
%         outref=complex(V_Fr,V_Fi);
%         outref(cp,cp,cp)=dcVal;
%         outref=smap.iftj(outref);
% 
% end;
% 
%     function outref=gpuInterp(x1i,x2i,x3i)
% %         x1i = (x1i-1)/ds+1;
% %         x2i = (x2i-1)/ds+1;
% %         x3i = (x3i-1)/ds+1;        
%         loc1 = min(Npix-1,floor(x1i));
%         loc2 = min(Npix-1,floor(x2i));
%         loc3 = min(Npix-1,floor(x3i));
%         
%         % x1i now becomes weight for x1i. Overwrite instead of creating new
%         % variable for memory performance. Same for x2i and x3i.
%         x1i = x1i-loc1;
%         x2i = x2i-loc2;
%         x3i = x3i-loc3;
% 
%         outref  = (1-x1i)*((1-x2i)*((1-x3i)*V(loc1  +(loc2-1)*Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1  +(loc2-1)*Npix+(loc3)  *Npix*Npix))  ...
%             +    x2i *((1-x3i)*V(loc1  +(loc2)  *Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1  +(loc2)  *Npix+(loc3)  *Npix*Npix))) ...
%             +    x1i *((1-x2i)*((1-x3i)*V(loc1+1+(loc2-1)*Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1+1+(loc2-1)*Npix+(loc3)  *Npix*Npix))  ...
%             +    x2i *((1-x3i)*V(loc1+1+(loc2)  *Npix+(loc3-1)*Npix*Npix)   ...
%             +    x3i *V(loc1+1+(loc2)  *Npix+(loc3)  *Npix*Npix)));
%         
%     end
%         
% 
% end
% 
