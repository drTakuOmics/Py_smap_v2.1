function [thr_ref,prec,recall,thr_F1,F1_opt]=PR_quick(tableref1,varargin);

prec_thrs=[0.99 0.95 0.9 0.85]; thr_F1=[];
mode='standard';
if( nargin>1 )
    if( ~isempty(varargin{1}) )
        prec_thrs=varargin{1};
    end;
    if( nargin>2 )
        mode=varargin{2};
    end;
else

end;

tableref=tableref1;

switch mode
    case 'standard'
        vals=tableref.peak_part; vals_ctrl=tableref.peak_part_ctrl;
    case 'opt'
        vals=tableref.peak_part_opt; vals_ctrl=tableref.peak_part_opt_ctrl;
end;

val_min=min([vals; vals_ctrl]);
val_max=max([vals; vals_ctrl]);
    
xx=linspace(val_min,val_max,5e3);
peaks=hist(vals,xx);
peaks_ctrl=hist(vals_ctrl,xx);

cs=sum(peaks)-cumsum(peaks);
cs_ctrl=sum(peaks_ctrl)-cumsum(peaks_ctrl);

prec=cs./(cs+cs_ctrl);
recall=cs./max(cs);
plot(recall,prec);
thr_ref=[];
for i=1:length(prec_thrs)
    try
        thr_ref(i)=xx(find(prec>prec_thrs(i),1,'first'));
    catch
        fprintf('no thr. found for precision %3.2f\n',prec_thrs(i));
    end;
end;
if( nargin>1 )
    hold on;
    plot(cs./max(cs),cs./(cs+cs_ctrl));
end;
grid on;
xlabel('est. recall'); ylabel('est. precision');

F1=2./((1./recall)+(1./prec));
thr_F1=xx(find(F1==max(F1),1,'first'));
F1_opt=max(F1);
