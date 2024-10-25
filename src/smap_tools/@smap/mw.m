function mw(map,filename,rez);
% wrapper that calls WriteMRC (Sigworth's program) 
% but uses the same input sequence as tw.m
% also transposes the image before writing. This undoes the transpose at
% read-in that corrects the reader's orientation swap. At the end of the
% day, reading in an .mrc file and writing it to a new file (no transforms
% at the command line or in matlab) generates an identical copy according
% to the fiji reader.
%
% function mw(map,filename,rez);

newMap=zeros(size(map,2),size(map,1),size(map,3),'single');
for i=1:size(map,3);
    newMap(:,:,i)=map(:,:,i)';
end;

map=newMap;

smap.WriteMRC(map,rez,filename)

