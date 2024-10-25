function [outref,varargout]=ri(fn,varargin);

try
    [pathstr,name,ext]=fileparts(fn);
    % [pathstr,name,ext]=fileparts(fn);
    try
        switch ext
            case '.tif'
                outref=smap.tr(fn);
            case '.mrc'
                [outref,rez]=smap.mr(fn);
                varargout{1}=rez;
            case '.dm4'
                [outref,sx,units]=smap.ReadDMFile(fn);
                varargout{1}=sx; varargout{2}=units;
            case '.img'
                outref=[]; fprintf('Imagic readfile not implemented yet\n');
            otherwise
                fprintf('%s file type is unknown...\n',ext);
        end;
    catch
        fprintf('Error opening file %s...\n',fn);
        outref=[];
    end;
    
catch
    disp('extension not recognized');
    outref=[];
end;



% [pathstr,name,ext]=fileparts(fn);
% if( isempty(ext) )
%     fns=dir([pathstr '/*SP*']);
%     [~,~,ext]=fileparts(fns(1).name);
%     fn=[fn ext];
% end;
% try
%     switch ext
%         case '.tif'
%             outref=smap.tr(fn);
%         case '.mrc'
%             [outref,rez]=smap.mr(fn);
%             varargout{1}=rez;
%         case '.dm4'
%             [outref,sx,units]=smap.ReadDMFile(fn);
%             varargout{1}=sx; varargout{2}=units;
%         case '.img'
%             outref=[]; fprintf('Imagic readfile not implemented yet\n');
%         otherwise
%             fprintf('%s file type is unknown...\n',ext);
%     end;
% catch
%     fprintf('Error opening file %s...\n',fn);
% end;
% 
% 

