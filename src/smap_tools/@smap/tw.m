function tw(m,fn,bps);

if( bps==16 )
    fmt=Tiff.SampleFormat.UInt;
elseif( bps==32 )
    fmt=Tiff.SampleFormat.IEEEFP;
end;

%fn=strrep(fn,'~','/Users/rickgauerj');
fn=strrep(fn,'~','/groups/denk/home/rickgauerj');
t=Tiff(fn,'w');
%t=Tiff(fn,'a');
tagstruct.ImageLength = size(m,1);
tagstruct.ImageWidth = size(m,2);
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = bps;
tagstruct.SamplesPerPixel = 1;
tagstruct.RowsPerStrip = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Software = 'MATLAB';
tagstruct.SampleFormat=fmt;%Tiff.SampleFormat.IEEEFP;
t.setTag(tagstruct);
t.write(m(:,:,1));
t.close();

if( size(m,3)>1 )
    for j=2:size(m,3)
        t=Tiff(fn,'a');
        tagstruct.ImageLength = size(m,1);
        tagstruct.ImageWidth = size(m,2);
        tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        tagstruct.BitsPerSample = bps;
        tagstruct.SamplesPerPixel = 1;
        tagstruct.RowsPerStrip = 16;
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        tagstruct.Software = 'MATLAB';
        tagstruct.SampleFormat=fmt;%Tiff.SampleFormat.IEEEFP;
        t.setTag(tagstruct);
        t.write(m(:,:,j));
%         fprintf('%d\n',j);
        t.close();
    end;
end;

