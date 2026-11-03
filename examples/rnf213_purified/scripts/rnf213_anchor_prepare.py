#!/usr/bin/env python3
"""Prepare a RNF213 anchor with the E3 region omitted, using the working 3.8° tile.

Default omission: RNF213 author residues 3589--4926 (E3) and all non-A chains.
--omit-ctd additionally holds out author residues 4927--5148, retaining modeled
N-arm, linker, AAA core and hinge. This edits the COMPUTATIONAL search template,
not the specimen. No Docker/GPU search is started.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import mrcfile


ATOM_FIELDS=['group_PDB','id','type_symbol','label_atom_id','label_alt_id',
    'label_comp_id','label_asym_id','label_entity_id','label_seq_id','pdbx_PDB_ins_code',
    'Cartn_x','Cartn_y','Cartn_z','occupancy','B_iso_or_equiv','pdbx_formal_charge',
    'auth_seq_id','auth_comp_id','auth_asym_id','auth_atom_id','pdbx_PDB_model_num']


def params(path):
    p={}
    for line in path.read_text().splitlines():
        fields=line.split('#',1)[0].split(maxsplit=1)
        if len(fields)==2:
            if fields[0] in p:
                raise ValueError(f'Duplicate parameter: {fields[0]}')
            p[fields[0]]=fields[1]
    return p


def host_path(value,root):
    if value.startswith('/work/'):
        return root/value[6:]
    path=Path(value)
    return path if path.is_absolute() else root/path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def selected_atom(row,omit_ctd):
    if row[0]!='ATOM' or row[18]!='A' or row[20]!='1':
        return False
    residue=int(row[16])
    if not 1<=residue<=5148 or 3589<=residue<=4926:
        return False
    return not (omit_ctd and residue>=4927)


def domain(residue):
    if residue<=1290:return 'N-arm'
    if residue<=1774:return 'Linker'
    if residue<=3405:return 'AAA core'
    if residue<=3588:return 'Hinge'
    if residue<=4926:return 'E3 module'
    return 'CTD'


def export_cif(records):
    text='data_RNF213_anchor_noE3\n#\nloop_\n'
    text+='\n'.join('_atom_site.'+field for field in ATOM_FIELDS)+'\n'
    return text+'\n'.join(' '.join(row) for row in records)+'\n#\n'


def updated_parameters(text,replacements):
    removed={'MW_target','PDBFile','chains','units','rotationsFile'}
    lines=[];seen=set()
    for line in text.splitlines():
        fields=line.split('#',1)[0].split()
        key=fields[0] if fields else ''
        if key in removed:continue
        if key in replacements:
            if key in seen:raise ValueError(f'Duplicate parameter: {key}')
            line=key+' '+replacements[key];seen.add(key)
        lines.append(line)
    lines.extend(key+' '+value for key,value in replacements.items() if key not in seen)
    return '\n'.join(lines)+'\n'


def preview(path,records,omit_ctd):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ca=[r for r in records if r[0]=='ATOM' and r[20]=='1' and r[3]=='CA' and r[18] in ('A','B')]
    xyz=np.array([[float(r[j]) for j in (10,11,12)] for r in ca])
    anchor=np.array([selected_atom(r,omit_ctd) for r in ca])
    xyz=(xyz-xyz[anchor].mean(axis=0))/10
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(xyz[~anchor,i],xyz[~anchor,j],s=5,color='#a77bcc',alpha=.5,
                   label='Held out: E3 / CTD / E2' if omit_ctd else 'Held out: E3 / E2')
        ax.scatter(xyz[anchor,i],xyz[anchor,j],s=5,color='#28a8b5',alpha=.7,label='Search anchor')
        ax.set(xlabel='XYZ'[i]+' (nm)',ylabel='XYZ'[j]+' (nm)',aspect='equal')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('RNF213 computational template · retained and held-out Cα atoms')
    fig.savefig(path,dpi=220);plt.close(fig)


def main():
    import gemmi
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--structure',default='7OIK.cif')
    p.add_argument('--source-tile',default='tiles_3p8/tile_0003')
    p.add_argument('--omit-ctd',action='store_true')
    p.add_argument('--out',default='anchor_noE3_pilot')
    args=p.parse_args()
    root=Path.cwd().resolve();out=Path(args.out)
    if out.is_absolute() or '..' in out.parts or out==Path('.'):
        p.error('--out must be a new project-relative folder.')
    out=root/out
    if out.exists():p.error(f'{out} already exists; previous searches are preserved.')
    source=Path(args.source_tile).resolve();structure=Path(args.structure).resolve()
    if 'Done combining search output' not in (source/'run.log').read_text():
        raise ValueError('Source full-template tile must be completed.')
    source_params=params(source/'search.par')
    if source_params['function']!='search_global' or int(source_params['nCores'])!=1:
        raise ValueError('Expected the working single-worker global search.')
    if not all(np.isclose(float(source_params[k]),3.8) for k in ('angle_inc','psi_inc')):
        raise ValueError('Use a completed 3.8° source tile.')
    image=host_path(source_params['imageFile'],root)
    with mrcfile.open(image) as m:
        if m.data.ndim!=2 or m.data.shape!=(1024,1024):
            raise ValueError('Expected the working 1024-pixel tile.')
        apix=float(source_params['aPerPix'])
        if not np.isclose(float(m.voxel_size.x),apix):
            raise ValueError('Image header and search pixel size differ.')
    table=gemmi.cif.read_file(str(structure)).sole_block().find('_atom_site.',ATOM_FIELDS)
    records=[list(row) for row in table]
    selected=[row for row in records if selected_atom(row,args.omit_ctd)]
    if len(selected)<100 or not any(r[18]=='A' and 3589<=int(r[16])<=4926
                                   for r in records if r[0]=='ATOM' and r[20]=='1'):
        raise ValueError('The source must contain RNF213 chain A, including its E3 module.')
    if len({row[1] for row in selected})!=len(selected):
        raise ValueError('Repeated retained atom IDs.')
    text=export_cif(selected)
    check=[list(row) for row in gemmi.cif.read_string(text).sole_block().find('_atom_site.',ATOM_FIELDS)]
    if check!=selected:
        raise ValueError('Exported CIF changed its atom records.')
    model_name='RNF213_anchor_noE3'+('_noCTD' if args.omit_ctd else '')+'.cif'
    docker_folder='/work/'+out.relative_to(root).as_posix()
    replacements=dict(imageFile=source_params['imageFile'],
        structureFile=docker_folder+'/'+model_name,modelFile=docker_folder+'/'+model_name,
        outputDir=docker_folder+'/output',angle_inc='3.8',psi_inc='3.8')
    config=updated_parameters((source/'search.par').read_text(),replacements)
    counts=Counter(domain(int(row[16])) for row in selected)
    parent=[row for row in records if row[0]=='ATOM' and row[18]=='A' and row[20]=='1']
    mass=len(selected)*13.824/1000
    manifest=dict(template_type='RNF213 domain-omission anchor; not a spherical fragment',
        source_CIF=str(structure),source_CIF_sha256=digest(structure),
        source_full_template_section=source.relative_to(root).as_posix(),
        source_full_template_parameters=source_params,source_image_sha256=digest(image),
        fragment_CIF=model_name,approximate_fragment_mass_kDa=mass,
        approximate_parent_RNF213_mass_kDa=len(parent)*13.824/1000,
        retained_atoms=len(selected),retained_atom_fraction=len(selected)/len(parent),
        atom_counts_by_domain=dict(counts),retained_author_chain='A',
        omitted_E3_author_residues=[3589,4926],
        omitted_CTD_author_residues=[4927,5148] if args.omit_ctd else [],
        excluded_other_chains=True,pixel_size_A=apix,angular_spacing_deg=3.8,
        final_detection_SNR_cutoff=8,
        mass_convention='Modeled retained atom count times 13.824 Da, following SMAP; not measured molecular weight',
        coordinates='Original CIF axes, atom IDs, coordinates and B factors retained',
        domain_reference='https://elifesciences.org/articles/56185',
        purpose='Detect using retained structure; reconstruct experimental patches to evaluate held-out density',
        caveat='Domain omission reduces direct template bias but does not eliminate alignment/selection bias',
        pose_comparison='New SP is centered independently; register CIF/SP origins before comparing positions')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.anchor-prep-',dir=out.parent) as tmp:
        stage=Path(tmp)/'prepared';stage.mkdir();(stage/'output').mkdir()
        (stage/model_name).write_text(text);(stage/'search.par').write_text(config)
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        preview(stage/'anchor_preview.png',records,args.omit_ctd)
        stage.rename(out)
    print(f'Prepared: {out}')
    print(f'Retained modeled mass: approximately {mass:.2f} kDa ({len(selected)} atoms)')
    print('Retained domains: '+json.dumps(dict(counts)))
    print('Omitted: E3 residues 3589–4926'+('; CTD residues 4927–5148' if args.omit_ctd else '')+'; all non-A chains')
    print('Uses exactly the previous tile image and microscope/search parameters at 3.8°.')
    print(f'Preview: {out/"anchor_preview.png"}')
    print(f'Parameters: {out/"search.par"}')
    print('Prepared only. No Docker container or GPU search started.')


if __name__=='__main__':main()
