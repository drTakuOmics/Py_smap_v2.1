function FOVref=num_to_FOV(numref);
%%
% smap.FOV_to_num(T(end,:).FOV)
% function numref=FOV_to_num(FOVref);
FOV_str=num2str(numref);
% FOV_str=double2str(numref);

alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% FOVref=[FOV_str(2:7) '_' alphabet(str2num(FOV_str(8:9))) '_' FOV_str(10:13)];

theDate=datestr(datenum(2014,1,1,12,0,0)+str2num(FOV_str(1:4)),'mmDDYY');

FOVref=[theDate '_' alphabet(str2num(FOV_str(5:6))) '_' smap.zp(str2num(FOV_str(7:9)),4)];

