function [ai,al,av]=readDatFile(fn,varargin);



% fn='search_listAboveThreshold.dat';
fid=fopen(fn,'r');
temp=[];
temp=fread(fid,inf,'double');
fclose(fid);
ai=temp(1:3:end); 
al=temp(2:3:end); 
av=temp(3:3:end);
clear temp;


