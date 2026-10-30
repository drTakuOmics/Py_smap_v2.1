function s = nansum(x, dim)
if nargin < 2, s = sum(x(~isnan(x))); else, x(isnan(x)) = 0; s = sum(x, dim); end
end
