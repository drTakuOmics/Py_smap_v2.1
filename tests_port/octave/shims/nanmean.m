function m = nanmean(x, dim)
% Octave shim for MATLAB nanmean (removed from recent Octave statistics pkg)
if nargin < 2, x = x(:); dim = 1; end
m = mean(x, dim, 'omitnan');
end
