function distInDeg=measureQD(q1,q2,varargin);
% % function distInDeg=measureQD(q1,q2,varargin);

% q1 is the single q
% q2 is the multiple q

if( (size(q1,1)~=4) | (isempty(strfind(class(q1),'double'))) )
    if( (size(q1,1)==3) )
        if( ~strcmp(class(q1),'gpuArray') )
            q1=normalizeRM(q1);
            q1=squeeze(double(quaternion.rotationmatrix(q1)));
        end;
    elseif( strcmp(class(q1),'quaternion') )
        q1=squeeze(double(q1));
    else
        %fprintf('format not recognized...\n');
    end;
end;
if( (size(q2,1)~=4) | (isempty(strfind(class(q2),'double'))) )
    if( (size(q2,1)==3) )
        if( ~strcmp(class(q2),'gpuArray') )
            q2=normalizeRM(q2);
            q2=squeeze(double(quaternion.rotationmatrix(q2)));
        end;
    elseif( strcmp(class(q2),'quaternion') )
        q2=squeeze(double(q2));
    else
        %fprintf('format not recognized...\n');
    end;
end;

%distInDeg=2.*real(acos(complex(abs(squeeze(sum(bsxfun(@times,q1,q2))))))).*180./pi;

pfact=2.*180./pi;
dq=arrayfun(@real,arrayfun(@acos,complex(arrayfun(@abs,sum(bsxfun(@times,q1,q2))))));
distInDeg=bsxfun(@times,pfact,dq);



% N=size(q2,3);
% 
% R_inv=inv(q1);
% distInDeg=nan(1,N);
% for i=1:N
%     r2r=q2(:,:,i)*R_inv;
%     distInDeg(i)=acos((trace(r2r)-1)/2);
% end;
% distInDeg=distInDeg.*180./pi;

% %% phased out 031417 for faster version using doubles
% function distInDeg=measureQD(q1,q2,varargin);
% % % function distInDeg=measureQD(q1,q2,varargin);
% % q1 is the single q
% % q2 is the multiple q
% % qOp (varargin{1}) operates on q1
% % takes ~0.18 seconds to calculate distInDeg between a 1x24 q and a 1x10000
% % matrix
% 
% 
% qOp=quaternion.eye(1);
% if( nargin>2 )    
%     qOp=varargin{1};
% end;
% 
% % if( ~strcmp(class(q1),'quaternion') )
% %     q1=quaternion.rotationmatrix(q1);
% % end;
% % if( ~strcmp(class(q2),'quaternion') )
% %     q2=quaternion.rotationmatrix(q2);
% % end;
% 
% distInDeg=zeros(length(qOp),length(q2));
% % % tic;
% for j=1:length(qOp)
%     distInDeg(j,:)=2.*real(acos(abs(dot(q1*qOp(j),q2)))).*180./pi;
% end;
% % tCalc=toc
% % q1=double(q1);
% % q2=double(q2);
% % distInDeg=2.*real(acos(abs(squeeze(sum(bsxfun(@times,q1,q2)))))).*180./pi;
% 
% % % slow:
% % % takes ~9 seconds to calculate distInDeg between 1 q and a 24x10000
% % % matrix
% % 
% % if( size(q1,2)>1 )
% %     qNew_1=quaternion.zeros(1);
% %     for i=1:length(q1)
% %         qNew_1=[qNew_1 repmat(q1(i),1,length(q2))];
% %     end;
% %     qNew_1=qNew_1(2:end);
% %     qNew_2=repmat(q2,1,size(q1,2));
% % else
% %     qNew_1=q1;
% %     qNew_2=q2;
% % end;
% %     
% % % distInDeg=2.*real(acos(abs(dot(q1,q2)))).*180./pi;
% % distInDeg=2.*real(acos(abs(dot(qNew_1,qNew_2)))).*180./pi;
% % 
% % if( size(q1,2)>1 )
% %     distInDeg=reshape(distInDeg,size(q2,2),size(q1,2));
% % end;
