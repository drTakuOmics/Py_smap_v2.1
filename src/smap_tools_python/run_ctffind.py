import subprocess
from pathlib import Path


def run_ctffind(obj):
    """Run the external *ctffind* program using parameters from ``obj``.

    The input ``obj`` is expected to be a mapping or object with ``CTF`` and
    ``proc`` attributes/keys mirroring the MATLAB structure. The function writes
    the parameter file, executes ``ctffind`` and parses the diagnostic output to
    populate ``obj['final']`` and ``obj['ID']`` entries.
    """
    # support both attribute and dict style access
    ctf_params = getattr(obj, "CTF", obj["CTF"])
    proc = getattr(obj, "proc", obj["proc"])
    out = getattr(obj, "final", obj.setdefault("final", {}))
    ident = getattr(obj, "ID", obj.setdefault("ID", {}))

    # write the parameter file expected by ctffind
    params = {}
    for key, val in ctf_params.items():
        params[key] = str(val)
    base = Path(proc["fullSum_image"])
    fn_out = base.with_name(base.stem + "_CTFFind_input.txt")
    with open(fn_out, "w") as fh:
        for k, v in params.items():
            fh.write(f"{k} {v}\n")

    # execute ctffind
    subprocess.run(["ctffind", str(fn_out)], check=True)

    # read diagnostic output
    diag_base = Path(ctf_params["output_diag_filename"])
    diag_fn = diag_base.with_suffix(".txt")
    with open(diag_fn, "r") as fh:
        lines = fh.readlines()[5:12]
    vals = [float(line.split()[0]) for line in lines[:7]]

    out["df1"] = vals[1] / 10.0
    out["df2"] = vals[2] / 10.0
    out["ast"] = vals[3] * 3.141592653589793 / 180.0
    ident["CTF"] = 1
    return obj

__all__ = ["run_ctffind"]
