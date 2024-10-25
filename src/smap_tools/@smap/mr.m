function [map,rez]=mr(filename,startSlice,numSlices,varargin);
% wrapper that calls ReadMRC (Sigworth's program)
% then transforms it so that it is again at the correct orientation.
% The expected orientation is the one that matches what we expect to see after
% converting with mrc2tif, reading in with tr(), and then doing a flipud on
% the image. Here, the operation that yields the same image is a transpose.
%
% also truncates the reported resolution to %7.4f because we are reasonable
% people
%
% [map,rez]=mr(filename,startSlice,numSlices,test);

if nargin<2
    startSlice=1;
end;
if nargin<3
    numSlices=inf;
end;
if nargin<4
    test=0;
else
    test=varargin{1};
end;

[map,s,mi,ma,av]=smap.ReadMRC(filename,startSlice,numSlices,test);

rez=str2num(sprintf('%7.4f',s.rez));
newMap=zeros(size(map,2),size(map,1),size(map,3),'single');
for i=1:size(map,3);
    newMap(:,:,i)=map(:,:,i)';
end;

map=newMap;


