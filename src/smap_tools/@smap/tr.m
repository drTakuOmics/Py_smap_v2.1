function outref=tr(fn);
if( isempty(findstr(fn,'.tif'))==1 )
    fn=[fn '.tif'];
end;

temp=imfinfo(fn);
w=temp(1).Width;
h=temp(1).Height;
nFrames=length(temp);
bps=temp(1).BitsPerSample;
if( bps==8 )
    fmt='uint8';
elseif( bps==16 )
    fmt='uint16';
elseif( bps==32 )
    fmt='single';
elseif( bps==64 )
    fmt='double';
end;

outref=zeros(h,w,nFrames,fmt);
t=Tiff(fn,'r');    
for i=1:nFrames
   t.setDirectory(i);
   outref(:,:,i)=t.read();
end
t.close();


