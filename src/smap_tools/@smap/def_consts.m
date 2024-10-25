function consts=def_consts();
% defines physical constants

consts=[];
consts.V=300e3;
% consts.m_e=1.587.*(9.109e-31); % at 300 kV
consts.m_e = 9.10938215e-31;
consts.h=6.636e-34;
consts.hbar=(6.636e-34)./(2.*pi);
consts.q_e=1.602e-19;
consts.wl=0.00197e-9; % wavelength at 300 kV
consts.IC=(2.*consts.m_e./(consts.hbar.^2)).*consts.q_e;
consts.k=2.*pi./consts.wl;
consts.Cs=2.7e-3;
consts.Cc=2.7e-3;
consts.a_i=0.05e-3; % Illumination aperture [rad]
consts.dE=0.7;
consts.c=2.99792458e8;
consts.c_v=consts.c;

% % function consts=def_consts();
% % % defines physical constants
% % %
% % % function consts=def_consts();
% % 
% % consts=[];
% % consts.m_e=9.109e-31;
% % consts.h=6.636e-34;
% % consts.q_e=1.602e-19;
% % consts.c_v=2.99792458e8;
% % consts.IC=(2.*consts.m_e./((consts.h./(2.*pi)).^2)).*consts.q_e;

