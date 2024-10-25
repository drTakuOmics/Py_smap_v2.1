function qOut=bumpQ(qIn,qBump);
% eliminates any uncertainty about q order of operations
% qIn is the original set of rotations
% qBump is 2nd set of rotations
% order is: qIn, then qBump
%

if( strcmp(class(qIn),'quaternion') )
    qIn=squeeze(RotationMatrix(qIn));
end;
if( strcmp(class(qBump),'quaternion') )
    qBump=squeeze(RotationMatrix(qBump));
end;

qOut=zeros(3,3,size(qBump,3));
ctr=1;
for i=1:size(qIn,3)
    for j=1:size(qBump,3)
        RM=qBump(:,:,j)*qIn(:,:,i);
        qOut(:,:,ctr)=RM;
        if( mod(ctr,5e3)==0 )
            disp(ctr);
        end;
        ctr=ctr+1;
    end;
end;
        
        
