function obj=preprocess(obj);

steps={'GC' 'MC' 'SF' 'CTF'};
calls={'gainCorr','motionCorr','sumFrames','runCTFFind'};
for i=1:length(steps)
    if( eval(['~getfield(obj.ID,' '''' char(steps{i}) '''' ')']) )
        eval(['obj=smap.' char(calls{i}) '(obj);']);
%         disp(['obj=smap.' char(calls{i}) '(obj);']);
    end;
end;

% obj=smap.motionCorr(obj);
% obj=smap.sumFrames(obj);
% obj=runCTFFind(obj);

