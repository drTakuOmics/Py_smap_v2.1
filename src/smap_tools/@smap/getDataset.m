function s=getDataset(datasetName,handles);

datasetsDir=smap.getPref(handles,'datasetsDir');
load([datasetsDir datasetName '.mat'],'s');

% dataID=evalin('base','dataID');
% data=evalin('base','data');
% inds=smap.parseCellArray(dataID,datasetName);
% 
% try
%     s=data{inds};
% catch
%     fprintf('Did not find dataset %s...\n',datasetName);
%     s=[];
% end;


