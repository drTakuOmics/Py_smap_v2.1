function qFig(fname,varargin);

h=gcf;
dims=[4 3];
if( nargin>1 )
    dims=varargin{1};
    
%     % Change size
%     set(h,'paperunits','inches')
%     set(h, 'PaperPositionMode', 'manual');
%     set(h,'papersize',[dims(1),dims(2)])
%     set(h,'paperposition',[0,0,dims(1),dims(2)])
    
else
    set(gcf,'paperposition',get(gcf,'position')/100);
end;
fs=9;
if( nargin>2 )
    fs=varargin{2};
end;

set(h, 'renderer', 'painters');

%ch=get(h,'Children');
ch=h.Children;

for i=1:length(ch)
    
    if( isempty(findstr(class(ch(i)),'Axes'))==0 )
        set(h,'CurrentAxes',ch(i));
        hh=gca;
        % Change font
        fontToUse='Arial';
        set(findall(hh,'-property','FontSize'),'FontSize',fs);%10)
        set(findall(hh,'-property','FontName'),'FontName',fontToUse)
        set(hh,'TickLength',[0.02 0.005]);
        set(hh,'XMinorTick','on');
        set(hh,'YMinorTick','on');
        set(hh,'XTickMode','manual');
        set(hh,'YTickMode','manual');
        set(hh,'TickDir','out');
        set(hh,'box','off');
%         if( i<length(ch) )
%             set(hh,'XTickLabel','');
%             set(hh,'YTickLabel','');
%         end;
        set(h,'Color',[1 1 1]);
        set(h,'inverthardcopy','off');
        
    end;
    
end;

if( isempty(findstr(fname,'.eps'))==1 )
    fname=[fname '.eps'];
end;
if( exist(fname,'file')==2 )
    delete(fname);
%     pause(5);
end;

fname_temp = strrep(fname,'.eps','_tmp.eps');
[baseDir_temp,f_temp] = fileparts(fname_temp);
fname_temp = ['~/' f_temp '.eps'];

% Export file
print('-depsc2',fname_temp)

% now read in the file
fid = fopen(fname_temp);
ff = char(fread(fid))';
fclose(fid);

%these are the only allowed fonts in MatLab and so we have to weed them out
%and replace them:
mlabfontlist = {'AvantGarde','Helvetica-Narrow','Times-Roman','Bookman',...
    'NewCenturySchlbk','ZapfChancery','Courier','Palatino','ZapfDingbats',...
    'Helvetica'};%,'Symbol'};

for k = 1:length(mlabfontlist)
    ff = strrep(ff,mlabfontlist{k},fontToUse);
end


% open the file up and overwrite it
fid = fopen(fname,'w');
fprintf(fid,'%s',ff);
fclose(fid);
delete(fname_temp);
