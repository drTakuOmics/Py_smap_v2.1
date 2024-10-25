function outref=getPref(handles,pref);

prefs=get(handles.prefs,'UserData');
delimiter=':';
if( strcmp(pref,'all')==0 )
    try
    temp=regexp(char(prefs(find((1-cellfun('isempty',strfind(cellstr(prefs),[pref delimiter]))),1,'first'))),[pref delimiter],'split');
    outref=temp{2};
    catch
        disp(['pref ' pref ' not found']);
        outref='';
    end;
else
    temp=char(prefs(find((1-cellfun('isempty',strfind(cellstr(prefs),delimiter))))));
    for j=1:size(temp,1)
        tempLine=regexp(temp(j,:),delimiter,'split');
        outref{j,1}=tempLine{1};
        outref{j,2}=tempLine{2};
    end;
end;

