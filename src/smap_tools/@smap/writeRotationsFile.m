%% 
function writeRotationsFile(R,varargin);
% write rotation matrices to an ASCII file
% 
% function outref=writeRotationsFile(R);

fn='testR.txt';
if( nargin>1 )
    fn=varargin{1};
end;

R_f=zeros(size(R),'single');
for i=1:size(R,3)
    R_f(:,:,i)=R(:,:,i)';
end;
RR=reshape(R_f,3,3*size(R_f,3));
temp=1:size(R,3);
temp=repmat(temp,3,1);
indexVector=reshape(temp,1,3*size(R,3));
fid=fopen(fn,'w');
fprintf(fid,'%7i\t%5.4f\t%5.4f\t%5.4f\r\n',cat(1,indexVector,RR));
fclose(fid);