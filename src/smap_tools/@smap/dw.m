function dw(inref,fn,varargin);


inref=gather(inref);
dims=size(inref);
inref=reshape(inref,1,prod(dims));

fid=fopen(fn,'w');
fwrite(fid,inref,'single');
fclose(fid);

fn_hdr=strrep(fn,'.dat','.hdr');
fid_hdr=fopen(fn_hdr,'w');
for i=1:length(dims)-1
    fprintf(fid_hdr,'%i\t',dims(i));
end;
fprintf(fid_hdr,'%i\n',dims(end));
fclose(fid_hdr);



