function [outref_RM, outref_EA] = calculate_search_grid(symmetry_symbol, angular_step_size, psi_step, varargin);
% adapted from cisTEM

%%

symmetry_number = 1;
phi_start = 0;
psi_max = 358;

symmetry_type = upper(symmetry_symbol(1));
if( length(symmetry_symbol)>1 )
    symmetry_number = str2num(symmetry_symbol(2:end));
end;
switch symmetry_type
    case 'C'
        phi_max = 360 / symmetry_number;
        theta_max = 90;
%         theta_max = 10
        test_mirror = 1;
    case 'D'
        phi_max = 360 / symmetry_number
        theta_max = 90;
        test_mirror = 0;
    case 'T'
        phi_max = 180;
        theta_max = 54.7;
        test_mirror = 0;
    case 'O'
        phi_max = 90;
        theta_max = 54.7;
        test_mirror = 0;
    case 'I'
        phi_max = 180;
        theta_max = 31.7;
        test_mirror = 0;
        
    otherwise
        
end

psi_vector = [0:psi_step:psi_max];

theta_max_local = theta_max;
phi_start_local = phi_start;

theta_step = theta_max / floor(theta_max / angular_step_size + 0.5);
% theta_start_local = abs(theta_step / 2 * randn(1));
theta_start_local = 0;

outref_EA = [];
for theta = theta_start_local:theta_step:(theta_max_local + theta_step / 2.0)
    number_of_search_positions = 0;
    if (theta == 0 || theta == 180)
        phi_step = phi_max;
    else
        %         // angular sampling was adapted from Spider subroutine VOEA (Paul Penczek)
        phi_step = abs(angular_step_size / sin(theta*pi/180));
        if (phi_step > phi_max), phi_step = phi_max;, end;
        phi_step = phi_max / floor(phi_max / phi_step + 0.5);
    end
    for (phi = 0:phi_step:phi_max)
        number_of_search_positions = number_of_search_positions + 1;
    end;
    number_of_search_positions = number_of_search_positions * length(psi_vector);
    
    phi_here = zeros(1,number_of_search_positions);
    theta_here = ones(1,number_of_search_positions).*theta;
    psi_here = zeros(1,number_of_search_positions);
    
    ctr = 1;
    for (phi = 0:phi_step:phi_max)
        for k = 1:length(psi_vector)
            phi_here(ctr) = phi + phi_start_local;
            psi_here(ctr) = psi_vector(k);
            ctr = ctr + 1;
        end;
    end;
    list_here = [phi_here; theta_here; psi_here];
    outref_EA = [outref_EA list_here];
end;
nRotations = size(outref_EA,2);

phi = outref_EA(1,:).*pi./180;
theta = outref_EA(2,:).*pi./180;
psi = outref_EA(3,:).*pi./180;

RM = zeros(3,3,nRotations);

c_phi = cos(phi);
c_psi = cos(psi);
c_theta = cos(theta);
s_phi = sin(phi);
s_psi = sin(psi);
s_theta = sin(theta);

RM(1,1,:)=squeeze( c_phi .* c_theta .* c_psi - s_phi .* s_psi );
RM(1,2,:)=squeeze( s_phi .* c_theta .* c_psi + c_phi .* s_psi );
RM(1,3,:)=squeeze( -s_theta .* c_psi );
RM(2,1,:)=squeeze( -c_phi .* c_theta .* s_psi - s_phi .* c_psi );
RM(2,2,:)=squeeze( -s_phi .* c_theta .* s_psi + c_phi .* c_psi );
RM(2,3,:)=squeeze( s_theta .* s_psi );
RM(3,1,:)=squeeze( s_theta .* c_phi );
RM(3,2,:)=squeeze( s_theta .* s_phi );
RM(3,3,:)=squeeze( c_theta );


% % C2xm: 1 & 2 look like flipuds (mirrored across horizontal)
if( test_mirror )
    % % % for Cn particles: put the symmetry axis along z (into/out of page)
    R_flip=RotationMatrix(quaternion.eulerangles('xyz',[180 0 0].*pi/180)); % C2x
%     R_flip=RotationMatrix(quaternion.eulerangles('xyz',[0 180 0].*pi/180)); % C2y
%     R_flip=RotationMatrix(quaternion.eulerangles('xyz',[0 0 180].*pi/180)); % C2z
    RM_mirror = zeros(3,3,size(RM,3));
    RM = smap.normalizeRM(RM);
    for j=1:size(RM,3)
        RM_mirror(:,:,j) = RM(:,:,j) * R_flip; % C2x
%         RM_mirror(:,:,j) = R_flip * RM(:,:,j); % C2xm
    end;
    RM_mirror = smap.normalizeRM(RM_mirror);
    outref_RM = cat(3,RM,RM_mirror);
else
    outref_RM = smap.normalizeRM(RM);
end;




% if( 0 )
% 
% %% test a symmetry-constrained rotation set
% 
% symm_type = 'C'
% % symm_type = 'D'
% 
% Nt = 4;
% 
% inc = 2.*pi./Nt;
% R_sym = zeros(3,3,Nt);
% % theta_here = 2.*pi./Nt;
% % R_sym(:,:,1) = [cos(theta_here) sin(theta_here) 0
% %         -sin(theta_here) cos(theta_here) 0
% %         0 0 1];
% % R_sym(:,:,1+Nt) = [cos(theta_here) -sin(theta_here) 0
% %         -sin(theta_here) -cos(theta_here) 0
% %         0 0 -1];
% % R_inv = [1     0     0
% %      0    -1     0
% %      0     0     -1];
% % 
% % for i=2:Nt
% %     R_sym(:,:,i)=R_sym(:,:,1)*R_sym(:,:,i-1);
% %     R_sym(:,:,i+Nt) = R_inv*R_sym(:,:,i);
% % end;
% 
% % %%
% for i=1:Nt
%     theta_here = ((i-1) .* inc);
%     R_sym(:,:,i) = [cos(theta_here) sin(theta_here) 0
%         -sin(theta_here) cos(theta_here) 0
%         0 0 1];
%     if( strcmp(symm_type,'D') )
%     R_sym(:,:,i+Nt) = [cos(theta_here) -sin(theta_here) 0
%         -sin(theta_here) -cos(theta_here) 0
%         0 0 -1];
%     end;
% end;
% 
% q_sym=quaternion.rotationmatrix(R_sym)
% smap.pairwiseQD(R_sym,R_sym)
% 
% R_sym = normalizeRM(R_sym);
% 
% % R=normalizeRM(readRotationsFile(['~/smap_ij/rotation/symm_C2x.txt']));
% R=normalizeRM(readRotationsFile(['~/smap_ij/rotation/symm_C4.txt']));
% 
% R_zero = R;
% 
% nRotations = size(R,3);
% 
% qt = smap.measureQD(R_zero(:,:,1), R_zero);
% qt = sort(qt, 'ascend');
% ep_q = qt(2)
% 
% R_test=normalizeRM(squeeze(RotationMatrix(quaternion.randRot(1,1000))));
% 
% % % Starting with the first and progressing one by one, multiply by the N-symmetry operators
% % % to make N new qs, then calculate the pairwise qds between those and the sorted set. 
% R_sym_dummy = zeros(size(R_sym));
% q2=squeeze(double(quaternion.rotationmatrix(R_zero)));
% pfact=2.*180./pi;
% qd_min = [];    
% for i=1:size(R_test,3)    
%     R_sym_here = R_sym_dummy;
%     R_init = R_test(:,:,i);
%     for j=1:(size(R_sym,3))
%         R_sym_here(:,:,j) = R_init * R_sym(:,:,j);
%     end;
%     d=zeros(size(R_sym,3),size(q2,2),'double');
%     for j=1:(size(R_sym,3))
%         d(j,:)=smap.measureQD(R_sym_here(:,:,j),q2);
%     end;
%     
%     qd_min(i) = min(d(:));
%     plot(qd_min,'x-'); drawnow;
% end;
% 
% 
%     
% end;
% 





%%
