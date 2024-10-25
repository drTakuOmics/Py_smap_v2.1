function outref=normalizeRM(inputRM,varargin);

for i=1:size(inputRM,3);
    T=inputRM(:,:,i);
    det(T);
    [U S V]=svd(T);
    Tcorrected = U*V'; % Drop the diagonal
    norm(T-Tcorrected);
    det(Tcorrected);
    outref(:,:,i)=Tcorrected;    
end;
