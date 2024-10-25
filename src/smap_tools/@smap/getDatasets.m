function smapProps = getDatasets(filename,varargin)

[~,~,raw]=xlsread(filename);
for i=1:size(raw,1)
    for j=1:size(raw,2)
        if( isnan(raw{i,j})==1 )
            raw{i,j}='';
        end;
    end;
end;

smapProps=raw;

