function varargout=gpuwhos;

z=evalin('caller','struct2cell(whos)'); 
zz=z(4,:);
inds=smap.parseCellArray(zz,'gpuArray'); 

n=z(1,inds);
if (~isempty(n))
    sumSize=0;
    for j=1:length(n)
        gpuVars(j).name=char(n{j});
        gpuVars(j).class=evalin('caller',['classUnderlying(' char(n{j}) ');']);
        temp=evalin('caller',['gather(' char(n{j}) ');']);
        eval(['temp=whos(''-var'',''temp'');']);
        gpuVars(j).size=temp.bytes;
        varList{j,1}=sprintf('%20s\t%12.0f\t%s', ...
            char(gpuVars(j).name),gpuVars(j).size,gpuVars(j).class);
        sumSize=sumSize+gpuVars(j).size;
    end;
    varList{j+1,1}=sprintf('total is %6.4f GB\n',sumSize./1e9);
    varList=char(varList);
    varargout{1}=varList;
    varargout{2}=gpuVars;
    varargout{3}=sumSize;
else
    varargout{1}=[];
end;


