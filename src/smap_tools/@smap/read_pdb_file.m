function varargout=read_pdb_file(fn,varargin);
% modified from fastPDBRead
%
% function varargout=read_pdb_file(fn,varargin);

FileID=fopen(fn);
rawText=fread(FileID,inf,'*char');
fclose(FileID);
splitLines=strread(rawText,'%s','delimiter','\n');
numLines=length(splitLines);

PDBdata.atomType = cell(1,numLines);
PDBdata.atomNum  = cell(1,numLines);
PDBdata.atomName = cell(1,numLines);
PDBdata.resName  = cell(1,numLines);
PDBdata.chain    = cell(1,numLines);
PDBdata.resNum   = cell(1,numLines);
PDBdata.X        = cell(1,numLines);
PDBdata.Y        = cell(1,numLines);
PDBdata.Z        = cell(1,numLines);
PDBdata.b_factor = cell(1,numLines);
PDBdata.comment  = cell(1,numLines);
PDBdata.filler=cell(1,numLines);
PDBdata.occ=cell(1,numLines);
PDBdata.line=cell(1,numLines);

ctr=1; hCtr=1;
header=[];
xn=[]; yn=[]; zn=[];
chainIDs=char; atomList=char;
bFactor=[]; inds=[]; headerInds=[]; resName=[]; occ=[];
lineType=[];
for n = 1:numLines
    thisLine = cell2mat(splitLines(n));
    if( length(thisLine)<78 )
        disp(['skipping ' thisLine]);
        thisLine=repmat(' ',1,78);
        splitLines{n}=[splitLines{n} repmat(' ',1,78-length(splitLines{n}))];
    end;
    PDBdata.line(n)  = {thisLine};
    
    if( ~isempty(strfind(thisLine(1:4),'ATOM'))||~isempty(strfind(thisLine(1:4),'TER'))||~isempty(strfind(thisLine(1:4),'HETA')) )
        try
            PDBdata.atomType(ctr) = {thisLine(1:6)};
            PDBdata.atomNum(ctr)  = {thisLine(7:11)};
            PDBdata.atomName(ctr) = {thisLine(12:16)};
            PDBdata.resName(ctr)  = {thisLine(17:20)};
            PDBdata.chain(ctr)    = {thisLine(21:22)};
            PDBdata.resNum(ctr)   = {thisLine(23:26)};
            PDBdata.filler(ctr)   = {thisLine(27:30)};
            PDBdata.X(ctr)        = {thisLine(31:38)};
            PDBdata.Y(ctr)        = {thisLine(39:46)};
            PDBdata.Z(ctr)        = {thisLine(47:54)};
            PDBdata.occ(ctr)      = {thisLine(55:60)};
            PDBdata.b_factor(ctr) = {thisLine(61:66)};
            PDBdata.comment(ctr)  = {thisLine(67:end)};
            xn(ctr)=str2num(char(PDBdata.X(ctr)));
            yn(ctr)=str2num(char(PDBdata.Y(ctr)));
            zn(ctr)=str2num(char(PDBdata.Z(ctr)));
            atomList{ctr}=strtrim(PDBdata.atomName(ctr));
            bFactor(ctr)=str2num(char(PDBdata.b_factor(ctr)));
            chainIDs(ctr)=strtrim(char(PDBdata.chain(ctr)));
            resName{ctr}=strtrim(char(PDBdata.resName(ctr)));
            occ(ctr)=str2num(char(PDBdata.occ(ctr)));
            inds(ctr)=n;
            ctr=ctr+1;
            lineType(n)=1;
            if( mod(n,1000)==0 )
                fprintf('%d\n',n);
            end;
        catch
            disp(['header at line ' num2str(n)]);
            headerInds(hCtr)=n;
            header{hCtr}=thisLine;
            hCtr=hCtr+1;
            lineType(n)=0;
        end;
    else
        headerInds(hCtr)=n;
        header{hCtr}=thisLine;
        hCtr=hCtr+1;
        lineType(n)=0;
    end;
end;

xyz=[xn; yn; zn];
[nx,ny]=find(isnan(xyz)==1);
if( ~isempty(ny) )
    xyz=xyz(:,1:ny(1)-1);
    length(find(isnan(xyz)))
end;

outvars={'xyz','atomList','bFactor','chainIDs','resName','occ','inds','header',...
    'headerInds','lineType','PDBdata','splitLines'};

for i=1:length(outvars)
    assignin('caller',outvars{i},eval(outvars{i}));
end;
