from smap_tools_python import fov_to_num, num_to_fov


def test_fov_roundtrip():
    ref = "030518_A_0123"
    

    


    assert num_to_fov(fov_to_num(ref)) == ref