function obj=runCTFFind(obj);

params=fieldnames(obj.CTF);
for i=1:length(params)
    temp=getfield(obj.CTF,params{i});
    if( isnumeric(temp) )
        temp=num2str(temp);
    end;
    vals{i}=temp;
end;
     
fnOut=strrep(obj.proc.fullSum_image,'.mrc','_CTFFind_input.txt');
fid=fopen(fnOut,'w');
for i=1:length(params)
    fprintf(fid,'%s %s\n',params{i},vals{i});
end;
fclose(fid);

words=['ctffind ' fnOut];
[~,z]=system(words);


fid=fopen(strrep(obj.CTF.output_diag_filename,'.mrc','.txt'),'r'); 
C=textscan(fid,'%s',inf,'HeaderLines',5); 
fclose(fid);
for i=1:7
    CTF(i)=str2num(C{1}{i});
end;


obj.final.df1=CTF(2)./10;
obj.final.df2=CTF(3)./10;
obj.final.ast=CTF(4).*pi./180;
obj.ID.CTF=1;

s=obj;
% save(['/tier2/denk/datasets/' obj.ID.ID '.mat'],'s');
save([smap.checkBaseDir 'datasets/' obj.ID.ID '.mat'],'s');

disp('done');

%%

