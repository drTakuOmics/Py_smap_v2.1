function putDataset(s,handles);

datasetsDir=smap.getPref(handles,'datasetsDir');
datasetName=s.prop.ID;

% dataID=evalin('base','dataID');
% data=evalin('base','data');
% inds=smap.parseCellArray(dataID,datasetName);
try
%     data{inds}=s;
%     assignin('base','data',data);
    save([datasetsDir datasetName '.mat'],'s');
catch
    fprintf('Did not find dataset %s...\n',datasetName);
end;


