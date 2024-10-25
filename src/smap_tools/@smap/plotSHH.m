function thr=plotSHH(bins,N,varargin); 

nfp=1;
if( nargin>2 )
    nfp=varargin{1};
end;

NN=sum(N,2);
Nsamples=sum(NN);
y=Nsamples-cumsum(NN);
x=bins;
YS=(erfc(bins./sqrt(2))./2).*Nsamples;
thr=sqrt(2).*erfcinv(nfp.*2./(Nsamples));

figure(404);
semilogy(bins,YS,'k--'); hold on;
semilogy(bins,Nsamples-cumsum(NN));
xlim([0 13]);
ylim([0.9 Nsamples*2]);
smap.avl(thr);
axis square; grid on;
xlabel('SNR');
ylabel('survival count');



