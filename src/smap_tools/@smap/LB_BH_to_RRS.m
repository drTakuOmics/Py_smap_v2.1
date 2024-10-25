function rrs_degref=LB_BH_to_RRS(q_LBref,q_BHref,varargin);

q_LB=q_LBref;
q_BH=q_BHref;

load('~/smap_ij/analysis/RRS_pca_030820.mat','coeff_b','mu_b','coeff_h','mu_h');
zero_LB=([zeros(1,3)]-mu_b)*coeff_b;
zero_BH=([zeros(1,3)]-mu_h)*coeff_h;

nRotations=size(q_LB,3);
rrs=zeros(nRotations,3);
for i=1:nRotations
    rv_LB=rotationMatrixToVector(q_LB(:,:,i));
    test=(rv_LB-mu_b)*coeff_b-zero_LB;
    coords_LB=test;
    
    rv_BH=rotationMatrixToVector(q_BH(:,:,i));
    test=(rv_BH-mu_h)*coeff_h-zero_BH;
    coords_BH=test;
    
    rrs(i,:)=[coords_LB(1:2) coords_BH(1)];

end;


rrs_degref=rrs.*(-180./pi);

% qb=quaternion([0.995780 -0.091771 0 0])
