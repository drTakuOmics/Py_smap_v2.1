function outref=checkBaseDir(varargin);
if( nargin>0 )
    inref=varargin{1};
else
    inref='';
end;

computerID=smap.whoami
switch computerID
    case '.B945E8C52A98'
        baseDir='Volumes/denk$';        
%         baseDirActual='tier2';
        baseDirActual='groups/denk/home/rickgauerj';
        if( ischar(inref) )
            fpath=fileparts(inref);
            temp=regexp(fpath,'/','split');
            if( length(temp)>1 )
                if( ~strcmp(temp{2},baseDir) )
                    outref=strrep(inref,baseDirActual,baseDir);
                else
                    outref=inref;
                end;
            else
                outref=['/' baseDir '/'];%['/' baseDir '/denk$/'];
            end;
        end;

    otherwise
        baseDir='groups/denk/home/rickgauerj';
        baseDirActual='groups/denk/home/rickgauerj';
        if( ischar(inref) )
            fpath=fileparts(inref);
            temp=regexp(fpath,'/','split');
            if( length(temp)>1 )
                if( ~strcmp(temp{2},baseDir) )
                    outref=strrep(inref,baseDirActual,baseDir);
                else
                    outref=inref;
                end;
            else
                outref=['/' baseDir '/'];
            end;
        end;

end;

% '/groups/denk/home/rickgauerj/'
% '/Volumes/denk$/'

