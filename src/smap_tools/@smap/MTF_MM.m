function F=MTF_MM(params,k);
%F=(params(1)./(1+params(4).*xdata.^2))+(params(2)./(1+params(5).*xdata.^2))+params(3);

nParams=length(params);
a=params(1:nParams/2);
lambda=params(nParams/2+1:end);
%xdata=xdata.*2;

F=sinc(k./2);
for i=1:length(a)
    F=F+a(i).*exp(-(pi.^2).*(lambda(i).^2).*(k.^2)./4);
end;



if 0
%%

%params0=[0.910214 0.047163 0.042623 0.888312 4.467267 57.442789]
params0=[-0.1366    0.0975    0.0391   -0.0002    1.7061   57.4421]; % these work
xdata=[0 0.25 0.03 0.5 1.0]
ydata=[1 0.90 0.96 0.78 0.50]
params=lsqcurvefit(@tempfun,params0,xdata,ydata)

%%
%params0=[0.910214 0.047163 0.042623 0.888312 4.467267 57.442789]; % from McMullan 2014
params=[-0.1366    0.0975    0.0391   -0.0002    1.7061   57.4421]; % these work
edgeSize=3710;
[k_2d]=smap.getKs(ones(edgeSize,edgeSize),0.5);
k=smap.radialmeanj(k_2d);
MTF_i=smap.MTF_MM(params,k);

%

end;
