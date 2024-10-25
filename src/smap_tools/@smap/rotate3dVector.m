function v_out=rotate3dVector(R,v);

if( size(v,1)~=3 )
    v_out=(R*v')';
else
    v_out=R*v;
end;

