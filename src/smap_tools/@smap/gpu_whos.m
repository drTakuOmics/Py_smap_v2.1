function gpu_whos(varargin);

%%
if( nargin>0 )
    A=varargin{1};
    
    sum_flag=0;
    if( nargin>1 )
        sum_flag=1;
    end;
    
    fprintf('\n');
    fprintf('on the GPU:\n');
    total_bytes=0;
    words_all=[]; word_ctr=1; bytes_all=[];
    for i=1:length(A)
        if( strcmp(A(i).class,'gpuArray') )
            complex_flag=0;
%             evalin('caller',['assignin(''caller'',''item_class'',classUnderlying(' A(i).name '))']);
            item_class=evalin('caller',['classUnderlying(' A(i).name ')']);
            item_size=A(i).size;
            switch item_class
                case {'single','int32','uint32'}
                    item_bytes=prod(item_size).*4;
                case {'double','int64','uint64'}
                    item_bytes=prod(item_size).*8;
                case {'char','logical'}
                    item_bytes=prod(item_size).*1;
                otherwise
                    item_bytes=0;
            end;
            evalin('caller',['assignin(''caller'',''complex_flag'',~isreal(' A(i).name '))']);
            complex_flag=evalin('caller',['~isreal(' A(i).name ')']);
            if( complex_flag )
                item_bytes=item_bytes.*2;
            end;
            bytes_all(word_ctr)=item_bytes;
            if( sum_flag )
%                 evalin('caller',['assignin(''caller'',''sum_val'',sum(abs(' A(i).name '(:)),''double''))']);
                sum_val=evalin('caller',['sum(abs(' A(i).name '(:)),''double'')']);
               sum_val=(gather(sum_val));
            else
                sum_val='000';
            end;
            total_bytes=total_bytes+item_bytes;
            item_name=sprintf('%30s',A(i).name);
            item_bytes=sprintf('%12s',num2str(item_bytes));
            item_size=sprintf('%15s',num2str(item_size));
            item_class=sprintf('%10s',item_class);
            if( complex_flag )
                item_class=[item_class '*'];
            end;
            item_sum=sprintf('%5.4d',(sum_val));
            words=[item_name item_bytes item_class '\t' item_size];
            if( sum_flag )
                words=[words '     ' item_sum];
            end;
            %fprintf([words '\n']);
            words_all{word_ctr}=words;
            word_ctr=word_ctr+1;
            
        end;
        
        if( strcmp(A(i).class,'cell') )
%             evalin('caller',['assignin(''caller'',''item_class_temp'',class(' A(i).name '{1}))']);
            item_class_temp=evalin('caller',['class(' A(i).name '{1})']);            
            if( strcmp(item_class_temp,'gpuArray') )
                for j=1:max(A(i).size)
                    try
%                     evalin('caller',['assignin(''caller'',''item_class'',classUnderlying(' A(i).name '{j}))']);
%                     evalin('caller',['assignin(''caller'',''item_size'',size(' A(i).name '{j}))']);
                    item_class=evalin('caller',['classUnderlying(' A(i).name '{j})']);
                    item_size=evalin('caller',['size(' A(i).name '{j})']);
                    switch item_class
                        case {'single','int32','uint32'}
                            item_bytes=prod(item_size).*4;
                        case {'double','int64','uint64'}
                            item_bytes=prod(item_size).*8;
                        case {'char','logical'}
                            item_bytes=prod(item_size).*1;
                        otherwise
                            item_bytes=0;
                    end;
%                     evalin('caller',['assignin(''caller'',''complex_flag'',~isreal(' A(i).name '{j}))']);
                    complex_flag=evalin('caller',['~isreal(' A(i).name '{j})']);
                    if( complex_flag )
                        item_bytes=item_bytes.*2;
                    end;
                    bytes_all(word_ctr)=item_bytes;
                    if( sum_flag )
%                         evalin('caller',['assignin(''caller'',''sum_val'',sum(abs(' A(i).name '{j}(:)),''double''))']);
                        sum_val=evalin('caller',['sum(abs(' A(i).name '{j}(:)),''double''))']);
                        sum_val=(gather(sum_val));
                    else
                        sum_val='000';
                    end;
                    total_bytes=total_bytes+item_bytes;
                    item_name=sprintf('%30s',A(i).name);
                    item_bytes=sprintf('%12s',num2str(item_bytes));
                    item_size=sprintf('%15s',num2str(item_size));
                    item_class=sprintf('%10s',item_class);
                    if( complex_flag )
                        item_class=[item_class '*'];
                    end;
                    item_sum=sprintf('%5.4d',(sum_val));
                    words=[item_name item_bytes item_class '\t' item_size];
                    if( sum_flag )
                        words=[words '     ' item_sum];
                    end;
                    %fprintf([words '\n']);
                    words_all{word_ctr}=words;
                    word_ctr=word_ctr+1;
                    catch
                        fprintf('missing part of %s\n', char(A(i).name));
                    end;
                end;
            end
        end;
        
    end;
    [~,sort_inds]=sort(bytes_all,'descend');
    bytes_all_cs=cumsum(bytes_all(sort_inds));
    for i=1:(word_ctr-1)
        fprintf([words_all{sort_inds(i)} '\n']);
    end;
    
    fprintf('total is %6.4f GB\n',total_bytes./1e9);
    
else
    disp('syntax is gpu_whos(whos)...');
end;




%%