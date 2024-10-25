function R=readRotationsFile(inref,varargin);
% read rotation matrices from an ASCII file
%
% function outref=readRotationsFile(inref);

fid=fopen(inref,'r');
A=fscanf(fid,'%f');
fclose(fid);
nRotations=size(A,1)./12;
inds=A(1:4:end);
A=A(setdiff(1:length(A),1:4:length(A)));
A=reshape(A,3,nRotations.*3);
R=reshape(A,3,3,nRotations);
for i=1:size(R,3)
    R(:,:,i)=R(:,:,i)';
end;
