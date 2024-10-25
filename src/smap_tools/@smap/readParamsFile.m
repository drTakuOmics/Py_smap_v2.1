function [paramsref,fn_type]=readParamsFile(fnref,varargin);
% function [paramsref,fn_type]=readParamsFile(fnref,varargin);
% read params file
fn_type='';
paramsref=[];
if( nargin>1 )
    fn_type=varargin{1};
else
    fid=fopen(fnref);
    lineCtr=1;
    while 1
        line = fgetl(fid);
        if ~ischar(line), break, end
        if( (isempty(strfind(line,'#')) ) & ~isempty(line) )
            lines_in{lineCtr}=line;
            lineCtr=lineCtr+1;
        end;
    end;
    fclose(fid);
    ind_fxn=find(~cellfun(@isempty,strfind(lines_in,'function')),1,'first');
    the_line=lines_in{ind_fxn};
    temp=split(the_line,'function ');
    fn_type=temp{2};
end;

switch fn_type
    case 'search_global'
        
        paramsref=struct('outputDir',[],'listFormat','dat','imageFile',[],'modelFile',[],'maskFile',[],'aPerPix',1.0,'defocus',[70 70 0], ...
            'MTF',[0 0.935 0 0 0.64],'arbThr',5.0,'highThr',[],'nCores',1,'units',[],'rotationsFile',[], ...
            'F_abs',0,'psdFilterFlag',1,'sectors',[],'Cs',0.0027,'Cc',0.0027,'V_acc',300000.0, ...
            'deltaE',0.7,'a_i',50e-6,'binRange',20,'nBins',1024,'aPerPix_search',[], ...
            'T_sample',100,'df_inc',50,'qThr',10,'dThr',10,'optimizeFlag',1,'optThr',7, ...
            'maskCrossFlag',1,'range_degrees',2.0,'inc_degrees',0.5,'debugFlag',0, ...
            'PDBFile',[],'MW_target',[], ...
            'bArb',0,'chains',[],'modelName',[],'edge_max',512,'defocus_format','CTFFind', ...
            'angle_inc',[],'psi_inc',[],'structureFile',[],'mask_params',[0 3],'keep_scratch_flag',0, ...
            'margin_pix',0);            
        
        fields=fieldnames(paramsref);
        
        for j=1:length(lines_in)
            for k=1:length(fields)
                z=regexp(lines_in{j},fields{k},'split');
                if( length(z)>1 )
                    zz=strtrim(z{2});
                    switch fields{k}
                        case {'outputDir','imageFile','modelFile','maskFile','rotationsFile','listFormat', ...
                                'PDBFile','chains','modelName','defocus_format','structureFile'} % string input
                            paramsref=setfield(paramsref,char(fields(k)),strtrim(zz));
                        otherwise % number input
                            the_number=str2num(strtrim(zz));
                            if( ~isempty(the_number) & isnumeric(the_number) )
                                    paramsref=setfield(paramsref,char(fields(k)),the_number);

                            end;
                    end;
                end;
            end;
        end;
        
    case 'search_local'
        
        paramsref=struct('outputDir',[],'modelFile',[],'maskFile',[],'aPerPix',1.0, ...
            'Cs',0.0027,'Cc',0.0027,'V_acc',300000.0,'deltaE',0.7,'a_i',50e-6, ...
            'rotationsFile',[], ...
            'tableFile',[],'mask_rad',[],'optimize_flag',1, ...
            'v1',[0 0 0],'the_q_corr',[1 0 0 0 1 0 0 0 1],'the_corr',[0 0 0], ...
            'ff_flag',1,'subtract_flag',0,'patch_edge',5,'subpixel_flag',0,'bArb',[], ...
            'template_edge',512,'q_offset',[1 0 0 0 1 0 0 0 1], ...
            'patchFile',[],'coordinateFile',[],'nCores',[],'thr',[],'keep_scratch_flag',0, ...
            'new_patches_flag',0);
        
        fields=fieldnames(paramsref);
        
        for j=1:length(lines_in)
            for k=1:length(fields)
                z=regexp(lines_in{j},fields{k},'split');
                if( length(z)>1 )
                    zz=strtrim(z{2});
                    switch fields{k}
                        case {'tableFile','outputDir','imageFile','modelFile','maskFile','rotationsFile','listFormat',...
                                'patchFile','coordinateFile'} % string input
                            paramsref=setfield(paramsref,char(fields(k)),strtrim(zz));
                        otherwise % number input
                            if( ~isempty(str2num(strtrim(zz))) )
                                paramsref=setfield(paramsref,char(fields(k)),str2num(strtrim(zz)));
                            else
                                paramsref=setfield(paramsref,char(fields(k)),strtrim(zz));
                            end;
                    end;
                end;
            end;
        end;
        
    case 'calculate_SP'
                
%         paramsref=struct('outputDir',[],'PDBFile',[],'aPerPix',1.0,'V_acc',300, ...
        paramsref=struct('outputDir',[],'PDBFile',[],'aPerPix',1.0,'V_acc',300000,'MW_target',[], ...
            'method','real','bArb',0,'nCores',1,'gpuFlag',1,'units',[],'chains',[],'modelName',[],'edge_max',512, ...
            'structureFile',[]);
        
        fields=fieldnames(paramsref);
        
        for j=1:length(lines_in)
            for k=1:length(fields)
                z=regexp(lines_in{j},fields{k},'split');
                if( length(z)>1 )
                    zz=strtrim(z{2});
                    if( strcmp(zz(end),'/') )
                        zz=zz(1:end-1);
                    end;
                    switch fields{k}
                        case {'outputDir','PDBFile','method','chains','modelName','structureFile'} % string input
                            paramsref=setfield(paramsref,char(fields(k)),strtrim(zz));
                        otherwise % number input
                            paramsref=setfield(paramsref,char(fields(k)),str2num(strtrim(zz)));
                    end;
                end;
            end;
        end;

        
        
    otherwise
        
        
        paramsref=[];
        
end;

%         fid=fopen(fnref);
%         lineCtr=1;
%         while 1
%             line = fgetl(fid);
%             if ~ischar(line), break, end
%             if( ~isempty(line) )
%                 if( isempty(strfind(line,'#')) )
%                     lines_in{lineCtr}=line;
%                     lineCtr=lineCtr+1;
%                 elseif( strfind(line,'#')>1 )
%                     eol=strfind(line,'#')-1;
%                     lines_in{lineCtr}=line(1:eol);
%                     lineCtr=lineCtr+1;
%                 end;
%             end;
%         end;
%         fclose(fid);
