function outref=searchForPDB(varargin);
% Opens a browser window at the RCSB to download .pdb or .gz files.
% Should give a new directory and entry in the table.
% % outref=searchForPDB(varargin);
% % input should be handles with user prefs (otherwise default directories
% are assumed)

handles=varargin{1};
downloadDir='~/Documents/MATLAB/';
modelsDir=[get(handles.userPrefsBox,'UserData') 'models/'];
% if( nargin>0 )
%     handles=varargin{1};
%     downloadDir=smap.getPref(handles,'downloadDir');
%     mapsDir=smap.getPref(handles,'mapsDir');
% end;
outref=[];

% open a web browser to rcsb to search for PDBs
url='http://www.rcsb.org/pdb/home/home.do';
% 'http://www.rcsb.org/pdb/files/1IES.pdb1.gz'
loc=[];
[stat,h,url]=web(url,'-new');
while( length(loc)==0 )
    pause(0.1);
    loc=get(h,'CurrentLocation');
end;
while(isempty(loc)==0)
    locLast=loc;
    loc=get(h,'CurrentLocation');
end;
url=locLast;
disp(url);
assignin('base','url',url);
fileParts=regexp(url,'structureId=','split');
rcsbBase=['http://www.rcsb.org/pdb/files/'];
if( length(fileParts)==2 )
    newPDB=fileParts{2};
    disp(['new PDB is ' newPDB]);
    
    %
    %
    
    % % make a getComputerProps method
%     cd(downloadDir);
%     z=dir([downloadDir '*.gz']);
% pause
    z=dir([downloadDir '*.gz']);
    if( length(z)==0 )
        [stat,h,url]=web([rcsbBase newPDB '.pdb1.gz']);
        z=dir([downloadDir '*.gz']);
        while( length(loc)==0 )
            pause(0.1);
            loc=get(h,'CurrentLocation');
        end;
        while(isempty(loc)==0)
            locLast=loc;
            loc=get(h,'CurrentLocation');
        end;
        url=locLast;
        disp(url);
        assignin('base','url',url);
        fileParts=regexp(url,'structureId=','split');
    end;
    
    for j=1:length(z)
        tempName=char(regexp(z(j).name,newPDB,'match','ignorecase'));
        if( length(tempName)>0 )
            zipName=z(j).name;
            %                         zipName=[tempName '.gz'];
            disp(['unzipping ' downloadDir zipName '...']);
            words=['gzip -d ' downloadDir zipName];
            system(words);
        end;
    end;
    
    z=dir([downloadDir '*.pdb*']); ctr=1;
    for j=1:length(z)
        tempName=char(regexp(z(j).name,newPDB,'match','ignorecase'));
        
        if( length(tempName)>0 )
            if(ctr>1)
                disp(['Enter to move ' tempName ' into structure dir ' modelsDir upper(tempName) '/ ...']);
            else
                disp(['Moving ' tempName ' into structure dir ' modelsDir upper(tempName) '/ ...']);
            end;
            if( exist([modelsDir upper(tempName) '/'],'dir')==7 )
                
            else
                disp(['making new structure dir for ' upper(tempName) '...']);
                mkdir([modelsDir upper(tempName) '/']);
            end;
            newStructure=[modelsDir upper(tempName) '/' upper(tempName) '.pdb'];
            movefile([downloadDir z(j).name],newStructure);
%             words=['mv ' downloadDir z(j).name ' ' newStructure]
%             system(words)
            outref=newStructure;
            ctr=ctr+1;
        end;
    end;
    
    %                 disp('no PDB downloaded');
    
else
    disp('last page was not a pdb download page');
end;
