function c = split(s, d)
% Octave shim for MATLAB split(str, delimiter) on a char row vector
c = strsplit(s, d)';
end
