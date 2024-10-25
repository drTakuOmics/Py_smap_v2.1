function SNR=estimate_SNR(MW_here,thickness,varargin);
% % 
% % function SNR=estimate_SNR(MW_here,thickness,varargin);
% % MW_here in kDa, thickness in nm
% % 

% beta=[21.2 1010];
% MW_ref=1950;
% fun=@(beta,thickness,MW_here)sqrt(MW_here./MW_ref).*beta(1).*(exp(-thickness./lambda_system)./ ...
%     (1 + exp(-thickness./lambda_system) - exp(-thickness./beta(2))).^0.5);

% beta=[21.92 366.2];
% MW_init=1950;
% 
% beta=[15.92 366.2]
% MW_init=800;
% 
% beta=[13.89 366.2]
% MW_init=600;
%
% 
% beta=[8.46 366.2];
% MW_init=200;

% beta=[11.38 366.2];
% MW_init=400;
% % % exponential (fits):
% fun=@(beta,thicknessref,MWref)sqrt(MWref./MW_init).*beta(1)./(exp(thicknessref./beta(2)));
% SNR=fun(beta,thickness,MW_here);
% plot(TT,183.13.*exp(2.*TT./366))

% 012420:
lambda=426;

% aa=[0.259 25.22];
% fun=@(lambda,thicknessref,MWref)sqrt(aa(1).*MWref+aa(2)).*exp(-thicknessref./lambda);
% SNR=fun(lambda,thickness,MW_here);

% % included an offset (2.0978) and a scale factor (0.4654) in the fit
% % both included in espected SNR estimate, which should apply to the normal usage
% % where a max step is involved:
aa=[0.4654 2.0978];
fun=@(lambda,thicknessref,MWref)(aa(2)+aa(1).*sqrt(MWref)).*(exp(-thicknessref./lambda));


% % included an offset (2.0978) and a scale factor (0.4654) in the fit
% % Including only the scale factor in SNR estimate; treats SNR as not involving a max step,
% % which is applicable to the ligand searches
% aa=[0.4654 2.0978];
% fun=@(lambda,thicknessref,MWref)(aa(1).*sqrt(MWref)).*(exp(-thicknessref./lambda));

% aa=[1.1229 0.4768]; % up to 1000 kDa
% fun=@(lambda,thicknessref,MWref)aa(1)+(aa(2).*sqrt(MWref)).*(exp(-thicknessref./lambda));

SNR=fun(lambda,thickness,MW_here);


