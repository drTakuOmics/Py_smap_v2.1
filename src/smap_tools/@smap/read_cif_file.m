function varargout=read_cif_file(params_cif);
%
% function varargout=read_cif_file(params_cif);

%ATOM   1     O  "O5'" . U   A  1  6    ? 133.894 111.247 1.629    1.00 75.85  ? 5    U   A "O5'" 1
%ATOM   1    N  N   . VAL A 1 1   ? 6.204   16.869  4.854   1.00 49.05 ? ? ? ? ? ? 1   VAL A N   1
%grep '^ATOM' 4HHB.cif | awk '{print $1, $3, $23, $24, $22, $11, $12, $13}'
% 2: atomNum
% 3: atomList
% 7 (19?): chainid
% 11, 12, 13: xyz
% 15: bFactor
%grep '^ATOM' 6hcf.cif | awk '{print $2, $3, $7, $11, $12, $13, $15}'


fn_cif=params_cif.CIFFile;
try
    if( ~isempty( params_cif.chains ) )
        words=['grep "^ATOM" ' fn_cif ' | '];
        words=[words 'awk -F '' '' ''$19 == "'];
        for i=1:length(params_cif.chains)
            chain_name=params_cif.chains{i};
            words=[words chain_name '"'];
            if( i<length(params_cif.chains) )
                words=[words ' || $19 == "'];
            else
                words=[words ''' | awk ''{print $2, $3, $19, $11, $12, $13, $15}'''];
            end;
        end;
        
    else
        words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $19, $11, $12, $13, $15}'''];
%         words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $13, $10, $11, $12, $17}''']; % for chimerax output
    end;
    disp(words);
    [~,resp]=system(words);
    
    split_entries=textscan(resp,'%s','delimiter',' ');
catch
    disp('Appears to be a non-traditional .cif format...');
    fn_cif=params_cif.CIFFile;
    if( ~isempty( params_cif.chains ) )
        words=['grep "^ATOM" ' fn_cif ' | '];
        words=[words 'awk -F '' '' ''$17 == "']; % for pymol output
        for i=1:length(params_cif.chains)
            chain_name=params_cif.chains{i};
            words=[words chain_name '"'];
            if( i<length(params_cif.chains) )
                %             words=[words ' || $19 == "'];
                words=[words ' || $17 == "']; % for pymol output
            else
                words=[words ''' | awk ''{print $2, $3, $17, $11, $12, $13, $15}''']; % for pymol output
            end;
        end;
        
    else
        %     words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $19, $11, $12, $13, $15}'''];
        words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $17, $11, $12, $13, $15}''']; % for pymol output
    end;
    disp(words);
    [~,resp]=system(words);
    
    split_entries=textscan(resp,'%s','delimiter',' ');
end;

atomNums=str2num(char(split_entries{1}(1:7:end)))';
atomList=split_entries{1}(2:7:end)';
chainIDs=split_entries{1}(3:7:end)';
xn=str2num(char(split_entries{1}(4:7:end)))';
yn=str2num(char(split_entries{1}(5:7:end)))';
zn=str2num(char(split_entries{1}(6:7:end)))';
bFactor=str2num(char(split_entries{1}(7:7:end)))';

% inds_to_use=find(cellfun(@isempty,strfind(atomList,'H')));
% atomNums=atomNums(inds_to_use);
% atomList=atomList(inds_to_use);
% chainIDs=chainIDs(inds_to_use);
% xn=xn(inds_to_use);
% yn=yn(inds_to_use);
% zn=zn(inds_to_use);
% bFactor=bFactor(inds_to_use);
% %pause;

xyz=[xn; yn; zn];
[nx,ny]=find(isnan(xyz)==1);
if( ~isempty(ny) )
    xyz=xyz(:,1:ny(1)-1);
    length(find(isnan(xyz)))
end;

outvars={'xyz','atomNums','atomList','bFactor','chainIDs'};

for i=1:length(outvars)
    assignin('caller',outvars{i},eval(outvars{i}));
end;


%%
% function varargout=read_cif_file(params_cif,varargin);
% %
% % function varargout=read_cif_file(params_cif,varargin);
% % params_cif.CIFFile = '~/smap_ij/gfish/CIF/mammal/9rsa.cif';
% % params_cif.chains={'A','C'};
% % [or will read all chains if empty or nonexistent]
% %
% 
% %ATOM   1     O  "O5'" . U   A  1  6    ? 133.894 111.247 1.629    1.00 75.85  ? 5    U   A "O5'" 1
% %ATOM   1    N  N   . VAL A 1 1   ? 6.204   16.869  4.854   1.00 49.05 ? ? ? ? ? ? 1   VAL A N   1
% %grep '^ATOM' 4HHB.cif | awk '{print $1, $3, $23, $24, $22, $11, $12, $13}'
% % 2: atomNum
% % 3: atomList
% % 7 (19?): chainid
% % 11, 12, 13: xyz
% % 15: bFactor
% % grep '^ATOM' 6hcf.cif | awk '{print $2, $3, $7, $11, $12, $13, $15}'
% 
% 
% fn_cif=params_cif.CIFFile;
% if( ~isempty( params_cif.chains ) )
%     words=['grep "^ATOM" ' fn_cif ' | '];
%     words=[words 'awk -F '' '' ''$19 == "'];
%     for i=1:length(params_cif.chains)
%         chain_name=params_cif.chains{i};
%         words=[words chain_name '"'];
%         if( i<length(params_cif.chains) )
%             words=[words ' || $19 == "'];
%         else
%             words=[words ''' | awk ''{print $2, $3, $19, $11, $12, $13, $15}'''];
%         end;
%     end;
% 
% else
%     words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $19, $11, $12, $13, $15}'''];
% end; 
% disp(words);
% [~,resp]=system(words);
% 
% if( isempty(resp) )
%     if( ~isempty( params_cif.chains ) )
%         words=['grep "^ATOM" ' fn_cif ' | '];
%         words=[words 'awk -F '' '' ''$17 == "']; % for pymol output
%         for i=1:length(params_cif.chains)
%             chain_name=params_cif.chains{i};
%             words=[words chain_name '"'];
%             if( i<length(params_cif.chains) )
%                 words=[words ' || $17 == "']; % for pymol output
%             else
%                 words=[words ''' | awk ''{print $2, $3, $17, $11, $12, $13, $15}''']; % for pymol output
%             end;
%         end;
%         
%     else
%         words=['grep "^ATOM" ' fn_cif ' | awk ''{print $2, $3, $17, $11, $12, $13, $15}''']; % for pymol output
%     end;
%     disp(words);
%     [~,resp]=system(words);
% 
% end;
% 
% split_entries=textscan(resp,'%s','delimiter',' ');
% atomNums=str2num(char(split_entries{1}(1:7:end)))';
% atomList=split_entries{1}(2:7:end)';
% chainIDs=split_entries{1}(3:7:end)';
% xn=str2num(char(split_entries{1}(4:7:end)))';
% yn=str2num(char(split_entries{1}(5:7:end)))';
% zn=str2num(char(split_entries{1}(6:7:end)))';
% bFactor=str2num(char(split_entries{1}(7:7:end)))';
% 
% xyz=[xn; yn; zn];
% [nx,ny]=find(isnan(xyz)==1);
% if( ~isempty(ny) )
%     xyz=xyz(:,1:ny(1)-1);
%     length(find(isnan(xyz)))
% end;
% 
% outvars={'xyz','atomNums','atomList','bFactor','chainIDs'};
% 
% for i=1:length(outvars)
%     assignin('caller',outvars{i},eval(outvars{i}));
% end;

