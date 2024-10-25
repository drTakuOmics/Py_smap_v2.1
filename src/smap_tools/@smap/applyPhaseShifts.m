function outref=applyPhaseShifts(inref,shifts,varargin);
% shift a matrix of values over by integer or subpixel distances
% shifts is a MxN vector with offsets in units of pixels
% gpuFlag=0;
% if( isa(inref,'gpuArray')==1 )
%     inref=gather(inref); gpuFlag=1; %wait(gdev);
% end;
ctu=class(inref);
if( isa(inref,'gpuArray') )
    sample=gather(inref(1,1,1)); %wait(gdev);
    complexFlag=~isreal(sample);
else
    complexFlag=~isreal(inref);
end;
% disp('test');


shifts_int=fix(shifts);
shifts_subpix=shifts-fix(shifts);

if( nargin<3 )
    cropFlag=0;
else
    cropFlag=varargin{1};
end;

if( (size(shifts,1).*size(shifts,2))==3 )
    volFlag=1;
else
    volFlag=0;
end;

if( volFlag==1 )
    movr=size(inref,1); movc=size(inref,2); movp=size(inref,3);
    xs=size(inref,1); ys=size(inref,2); zs=size(inref,3);
    [xd,yd,zd]=meshgrid(((0:xs-1)-xs/2)/xs,((0:ys-1)-ys/2)/ys,((0:zs-1)-zs/2)/zs);
    
    %pause;
    
    if( complexFlag==1 )
        if( max(abs(shifts_subpix))>0 ) 
            di_f=inref;            
            clear inref;
            dphs=yd.*(-shifts_subpix(1))+xd.*(-shifts_subpix(2))+zd.*(-shifts_subpix(3));
            clear xd yd zd;
            dphs=exp(1i.*2.*pi.*dphs);
            d_done=di_f.*dphs;
            movShifted=smap.iftj(d_done);
%            movShifted=fftshift(ifftn(ifftshift(d_done)));
        else
            movShifted=smap.iftj(inref);
%            movShifted=fftshift(ifftn(ifftshift(inref)));
        end;
        outref=circshift(movShifted,[shifts_int(1) shifts_int(2) shifts_int(3)]);
    else
        disp('applying 3D phase shifts to volume...');
        inref=circshift(inref,[shifts_int(1) shifts_int(2) shifts_int(3)]);
        if( max(abs(shifts_subpix))>0 )
            [xd,yd,zd]=meshgrid(((0:xs-1)-xs/2)/xs,((0:ys-1)-ys/2)/ys,((0:zs-1)-zs/2)/zs);            
            di_f=smap.ftj(inref);
            clear inref;
            dphs=yd.*(-shifts_subpix(1))+xd.*(-shifts_subpix(2))+zd.*(-shifts_subpix(3));
            clear xd yd zd;
            dphs=exp(1i.*2.*pi.*dphs);
            d_done=di_f.*dphs;
            mrDone=smap.iftj(d_done);
        else
            mrDone=inref;
        end;
        outref=mrDone;
    end;
    
else
    
    movNP=inref;
    nFrames=size(movNP,3);
    
    xs=size(inref,1); ys=size(inref,2);
    [xd,yd]=meshgrid(((0:xs-1)-xs/2)/xs,((0:ys-1)-ys/2)/ys);
    eval(['movShifted=zeros(xs,ys,nFrames,' '''' ctu '''' ');']);
    %     movShifted=zeros(xs,ys,nFrames);
    
    if( complexFlag==1 )
        for i=1:nFrames
            %if( max(abs(shifts_subpix(i,:)))>0 )
                di_f=movNP(:,:,i);%smap.ftj(movNP(:,:,i));
                dphs=xd'.*(-shifts(i,2))+yd'.*(-shifts(i,1));
                dphs=exp(1i.*2.*pi.*dphs);
                d_done=di_f.*dphs;
                movShifted(:,:,i)=smap.iftj(d_done);
            %else
            %    di_f=movNP(:,:,i);
            %    dphs=exp
            %    movShifted(:,:,i)=smap.iftj(movNP(:,:,i));
%                disp('flag')
            %end;
            %outref=circshift(movShifted(:,:,i),[shifts_int(i,2) shifts_int(i,1)]);
        end
        
    else
        
        for i=1:nFrames
            movNP(:,:,i)=circshift(movNP(:,:,i),[shifts_int(i,2) shifts_int(i,1)]);
            if( max(abs(shifts_subpix(i,:)))>0 )
                di_f=smap.ftj(movNP(:,:,i));
                dphs=xd'.*(-shifts_subpix(i,2))+yd'.*(-shifts_subpix(i,1));
                dphs=exp(1i.*2.*pi.*dphs);
                d_done=di_f.*dphs;
                movShifted(:,:,i)=smap.iftj(d_done);
            else
                movShifted(:,:,i)=movNP(:,:,i);
            end;
            if( cropFlag==1 )
                dummy=circshift(movNP_dummy,ceil([shifts(i,2),shifts(i,1)]));
                movShifted(:,:,i)=movShifted(:,:,i).*dummy;
            end;
        end;
        
        if( cropFlag==1 )
            movSum=sum(movShifted,3);
            badCols=find((nansum(movSum,1))==0);
            badRows=find((nansum(movSum,2))==0);
            goodRows=setdiff(1:size(movSum,1),badRows);
            goodCols=setdiff(1:size(movSum,2),badCols);
            movShifted=movShifted(goodRows,goodCols,:);
        end;
        
    end;
    outref=movShifted;
end;