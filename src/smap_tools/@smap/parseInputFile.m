function varargout=parseInputFile(obj);

for i=1:length(varargin)
    ptf{i}=varargin{i};
end;

nout=length(ptf);

a=[];
fid=fopen('sampleParams.txt','r');
lineCtr=1;
while 1
    line = fgetl(fid);
    if ~ischar(line), break, end
        if( (isempty(strfind(line,'#')) ) & ~isempty(line) )
            a{lineCtr}=line;
            lineCtr=lineCtr+1;
        end;
    end;
fclose(fid);

aa=[];
for i=1:length(a)
    line=(a{i});
    if( isempty(strfind(line(1),'#')) )
        for j=1:length(ptf)
            if( ~isempty(regexp(line,ptf{j},'match')) )
                aa{i}=line;
            end;
        end;

    end;    
end;

% pause;

for i=1:length(aa)
    line=aa{i};
    for j=1:length(ptf)
        if( ~isempty(regexp(line,ptf{j},'match')) )
            tempLine=char(ptf{j}(1));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                fn_image=temp{2};
                varargout{i}=fn_image;
            end;
            tempLine=char(ptf{j}(2));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                fn_template=temp{2};
                varargout{i}=fn_template;
            end;
            tempLine=char(ptf{j}(3));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                aPerPix=str2num(temp{2});
                varargout{i}=aPerPix;
            end;
            tempLine=char(ptf{j}(4));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                df=str2num(temp{2});
                varargout{i}=df;
            end;
            tempLine=char(ptf{j}(5));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                MTF=str2num(temp{2});
                varargout{i}=MTF;
            end;
            tempLine=char(ptf{j}(6));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                arbThr=str2num(temp{2});
                varargout{i}=arbThr;
            end;
            tempLine=char(ptf{j}(7));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                binRange=str2num(temp{2});
                varargout{i}=binRange;
            end;
            tempLine=char(ptf{j}(8));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                edgeSize=str2num(temp{2});
                varargout{i}=edgeSize;
            end;
            tempLine=char(ptf{j}(9));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                envFlag=str2num(temp{2});
                varargout{i}=envFlag;
            end;
            tempLine=char(ptf{j}(10));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                rotationsPerFile=str2num(temp{2});
                varargout{i}=rotationsPerFile;
            end;
            tempLine=char(ptf{j}(11));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                dot_mean=str2num(temp{2});
                varargout{i}=dot_mean;
            end;
            tempLine=char(ptf{j}(12));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                dot_std=str2num(temp{2});
                varargout{i}=dot_std;
            end;
            tempLine=char(ptf{j}(13));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                maskSize=str2num(temp{2});
                varargout{i}=maskSize;
            end;
            tempLine=char(ptf{j}(14));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                fn_rotations=temp{2};
                varargout{i}=fn_rotations;
            end;
            tempLine=char(ptf{j}(15));
            if(~isempty(strfind(line,[tempLine '='])))
                temp=regexp(line,[tempLine '='],'split');
                imageMask=str2num(temp{2});
                varargout{i}=imageMask;
            end;

        end;
    end;
end;

% for i=1:nout
%     varargout{i} = s(i);
% end

