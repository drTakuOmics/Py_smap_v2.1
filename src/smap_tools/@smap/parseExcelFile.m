function outref=parseExcelFile(varargin);

ID=[];
if( nargin>0 )
    ID=varargin{1};
end;


pause;
% % hard-coded:
datasetsDir='/Volumes/denk-1/datasets/';
% datasetsDir=smap.getPref(handles,'datasetsDir');
[~,~,smapList]=xlsread([datasetsDir 'smap_properties.xlsx']);

if( isempty(ID)==0 )
    propName={};
    for i=1:size(smapList,2)
        propName{i}=[char(smapList{1,i}) '.' char(smapList{2,i})];
    end;
    newInd=find(1-cellfun('isempty',strfind(cellstr([smapList(:,1)]),ID)));
    try
        for i=1:size(smapList,2)
            ii=smapList{newInd,i}; nanFlag=0;
            if( isnumeric(ii)==1 )
                eval(['obj.' char(propName{i}) '=' num2str(ii) ';']);
                if(strcmp(num2str(ii),'NaN')==1 )
                    nanFlag=1;
                end;
            else
                eval(['obj.' char(propName{i}) '=' '''' ii '''' ';']);
                if(strcmp(ii,'NaN')==1 )
                    nanFlag=1;
                end;
            end;
            if( nanFlag )
                eval(['obj.' char(propName{i}) '=' '''' '''' ';']);
            end;
        end;
        
    catch
        disp(['could not locate ID# ' ID]);
        obj=[];
    end;
    
else
    propName={};
    for i=1:size(smapList,2)
        if( sum(isnan((smapList{2,i})))==0 )
            propName{i}=[char(smapList{1,i}) '.' char(smapList{2,i})];
        else
            propName{i}=char(smapList{1,i});
        end;    
    end;
    
    try
        ctr=1;
        for j=1:size(smapList,1)
            for i=1:size(smapList,2)
                ii=smapList{j,i}; nanFlag=0;
                if( isnumeric(ii)==1 )
                    eval(['obj.' char(propName{i}) '=' num2str(ii) ';']);
                    if(strcmp(num2str(ii),'NaN')==1 )
                        nanFlag=1;
                    end;
                else
                    eval(['obj.' char(propName{i}) '=' '''' ii '''' ';']);
                    if(strcmp(ii,'NaN')==1 )
                        nanFlag=1;
                    end;
                end;
                if( nanFlag )
                    eval(['obj.' char(propName{i}) '=' '''' '''' ';']);
                end;
            end;
            newObj(ctr)=obj; ctr=ctr+1;
        end;
        obj=newObj;
    catch
        disp(['could not locate ID# ' ID]);
        obj=[];
    end;
    
    
end;

outref=obj;



