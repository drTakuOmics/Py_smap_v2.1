function obj=writeSearchParams(obj);

% fxnToRun=obj.params.method;
% objectFilepath=['/tier2/denk/searches/' obj.ID.ID '.mat'];
% coresPerJob=obj.params.nCores;
% searchDir=obj.results.searchDir;
% 
% % % make these not hard-coded:
% interval=1.0
% jobNums=1:5
% startTime=[22 44]
% 
% % % add this back in here for multiple submits: 
% 
% % for loopCtr=1:nSubmits
% loopCtr=1;
%     h_t=startTime(loopCtr,1); m_t=startTime(loopCtr,2);
%     waitFlag=1;
%     while( waitFlag==1 )
%         fprintf('%s\n',datestr(now,13));
%         temp=regexp(datestr(now,13),':','split');
%         h_c=str2double(temp(1)); m_c=str2double(temp(2));
%         if( h_c==h_t & m_c==m_t )
% %             cd(['~/temp/' targetDir{loopCtr}]);
%             cd(['/tier2/denk/']);            
%             
%             words={'. /sge/current/default/common/settings.sh'
%             
% 
%             fid=fopen([searchDir 'smap.log'],'w');
%             fprintf(fid,'%s\n',subWords);
%             fclose(fid);
%             disp(subWords);
%             waitFlag=0;
% 
%             for j=1:length(jobNums)
%                 currentNum=jobNums(j);                
%                 
%                 % % this needs to include an ssh to the open socket:
%                 
%                 disp('would submit here....')
% 
% %                 -o /dev/null -j y 
%                 
% %                 words=['ssh rickgauerj@login1 $qsub -pe batch ' ...
% %                     num2str(coresPerJob) ' -b y -N ' ...
% %                     'smap' num2str(currentNum) ' -cwd -V -l sandy=true ' '''/groups/denk/home/rickgauerj/./matlab_wrapper.sh ' ...
% %                     fxnToRun ' ' num2str(currentNum) ' ' objectFilepath ' > ' searchDir 'output' num2str(currentNum) '.txt' ''' '];
% 
%                 !ssh rickgauerj@login1
%                 
%                     
%                 words=['qsub -pe batch ' ...
%                     num2str(coresPerJob) ' -b y -N ' ...
%                     'smap' num2str(currentNum) ' -cwd -V -l sandy=true ' '''/groups/denk/home/rickgauerj/./matlab_wrapper.sh ' ...
%                     fxnToRun ' ' num2str(currentNum) ' ' objectFilepath ' > ' searchDir 'output' num2str(currentNum) '.txt' ''' '];
%                 
%                 eval(words)
%                 
%                 !exit
% 
% 
% %                 ssh rickgauerj@login1 qsub -pe batch 1 -b y -N smap1 -cwd -V -l sandy=true '/groups/denk/home/rickgauerj/./matlab_wrapper.sh smappoi 1 /tier2/denk/searches/100314_0001_APOR_0001-0001.mat > output1.txt' 
%                 
% %                 words=['ssh rickgauerj@login1 ls -lrth /tier2/denk/ > output' num2str(currentNum) '.txt'];
% 
% 
%                 logLine=sprintf('%s\n',words);
%                 fid=fopen([obj.results.searchDir 'smap.log'],'a');
%                 fprintf(fid,'%s\n',logLine);
%                 fclose(fid);
% 
%                 [s,w]=system(words);
%                 
%                 ww=regexp(w,'\s','split');
%                 jobIDTemp=[];
%                 for wwCtr=1:length(ww)
%                     jobIDTemp(wwCtr)=str2double(ww{wwCtr});
%                 end;
%                 jobID=num2str(nanmax([0 jobIDTemp]));
%                 logLine=sprintf('%s\t%s\t%s',num2str(currentNum),jobID,datestr(now,31));
%                 fid=fopen([obj.results.searchDir 'smap.log'],'a');
%                 fprintf(fid,'%s\n',logLine);
%                 fclose(fid);
%                 pause(interval);
%             end;
% 
%             break;
%         end;
% 
%         pause(30);
%     end;
% 
% % end;


