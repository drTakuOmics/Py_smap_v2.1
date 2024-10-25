function numref=FOV_to_num(FOVref);
%%
% smap.FOV_to_num(T(end,:).FOV)
% function numref=FOV_to_num(FOVref);

if( iscell(FOVref) )
    FOVref=char(FOVref);
end;

% blank_name=repmat('0',1,13);
% blank_name(1)='1';
blank_name=repmat('0',1,9);

alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ';



parts=regexp(FOVref,'_','split');
theYear=str2num(['20' parts{1}(5:6)]);
theMonth=str2num(parts{1}(1:2));
theDate=str2num(parts{1}(3:4));


temp=blank_name;
temp(1:4)=num2str(datenum(2018,3,5,12,0,0)-datenum(2014,1,1,12,0,0));
temp(1:4)=num2str(datenum(theYear,theMonth,theDate,12,0,0)-datenum(2014,1,1,12,0,0));
temp(5:6)=smap.zp(strfind(alphabet,char(parts{2})),2);
temp(7:9)=smap.zp(str2num(parts{3}),3);
numref=str2double(temp);

% temp(2:7)=char(parts{1});
% temp(8:9)=smap.zp(strfind(alphabet,char(parts{2})),2);
% temp(10:13)=smap.zp(char(parts{3}),4);
% numref=int64(str2num(temp));


