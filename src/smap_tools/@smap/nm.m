function outref=nm(inref);

x=inref(:);

nans = isnan(x);
x(nans) = 0;
n = sum(~nans,1);
% Sum up non-NaNs, and divide by the number of non-NaNs.
m = sum(x,1) ./ n;

denom = max(n-1, 1);
x0 = x - m;
y = sum(abs(x0(~nans)).^2,1) ./ denom; % abs guarantees a real result
ns=(y.^0.5);

outref=(inref-m)./ns;
% toc

% end;