function [inds,entries]=parseCellArray(inref,str);


inds=find(1-cellfun('isempty',strfind(cellstr(inref),char(str))));
% inds=find(1-cellfun('isempty',regexpi(cellstr(inref),char(str),'match')));
if( isempty(inds)==0 )
    for j=1:length(inds)
        entries{j}=inref{inds(j)};
    end;
else
    entries={};
end;

    
