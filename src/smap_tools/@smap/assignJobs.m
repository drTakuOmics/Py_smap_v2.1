function indsref=assignJobs(nIndices,nServers,serverID,varargin);
% calculate a vector of indices to share the work across servers
%
% function indsToUse=assignJobs(nIndices,nServers,serverID)

baseJobs=ceil(nIndices/nServers);
jobsPerServer=ones(1,nServers).*baseJobs;
jobsWithBase=cumsum(jobsPerServer);
startInds=jobsWithBase-baseJobs+1;
lastJobToUse=min([jobsWithBase(end) nIndices]);
jobsWithBase(end)=lastJobToUse;

indsToUse=[]; nJ=[];
for j=1:nServers
    indsToUse{j}=[startInds(j):jobsWithBase(j)];
    nJ(j)=length(indsToUse{j});
end;

indsref=indsToUse{serverID};
