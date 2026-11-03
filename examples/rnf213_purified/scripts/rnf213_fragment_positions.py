#!/usr/bin/env python3
"""Check partial/full-template correspondence on the SAME RNF213 image tile.

Uses saved experimental patches to register positions, checks the rotation
conversion against saved templates, then registers each CIF to its independently
centered SP model. No GPU searches or 3D reconstruction are launched.

Requires the existing rnf213_fragment_compare.py, rnf213_surface_overlay.py and
rnf213_workflow_v3.py beside this script. Positions are approximate original
pixel coordinates. Positional correspondence does not guarantee particle identity.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import mrcfile
from scipy.optimize import linear_sum_assignment

import rnf213_fragment_compare as compare
import rnf213_surface_overlay as overlay
import rnf213_workflow_v3 as workflow


def csv_write(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def load_sp(path,apix):
    with mrcfile.open(path) as m:
        volume=np.array(m.data,dtype=np.float32,copy=True)
        voxel=np.array([m.voxel_size.x,m.voxel_size.y,m.voxel_size.z],float)
        axes=tuple(int(m.header[k]) for k in ('mapc','mapr','maps'))
    if volume.ndim!=3 or len(set(volume.shape))!=1 or volume.shape[0]%2:
        raise ValueError('Expected a cubic, even SP grid.')
    if axes!=(1,2,3) or not np.allclose(voxel,apix,atol=1e-5):
        raise ValueError('SP axes or voxel size need explicit handling.')
    return volume


def sp_path(run,root):
    model=compare.host_path(run['params']['modelFile'],root)
    if model.suffix.lower()=='.mrc':
        return model
    candidate=run['output']/(model.stem+'_SP.mrc')
    if not candidate.is_file():
        choices=sorted(run['output'].glob('*_SP.mrc'))
        if len(choices)!=1:
            raise FileNotFoundError(f'Cannot select generated SP in {run["output"]}.')
        candidate=choices[0]
    return candidate


def register_positions(run,root):
    image=workflow.read_array(compare.host_path(run['params']['imageFile'],root))
    if image.ndim!=2:
        raise ValueError('Expected a two-dimensional tile image.')
    seeds=workflow.matrix(run['output']/'list_final.txt')
    patches=compare.read_stack(run['output']/'particles.mrc')
    if not (len(seeds)==len(patches)==len(run['scores'])):
        raise ValueError('Final lists and patch stacks differ in length.')
    with mrcfile.open(run['output']/'search_mean.mrc') as m:
        search_shape=np.array(m.data.shape[-2:])
    ratio=float(run['params']['aPerPix_search'])/float(run['params']['aPerPix'])
    center_search=search_shape//2+1
    center_original=np.array(image.shape)//2+1
    records=[]
    for i,(seed,patch,score,angle) in enumerate(zip(seeds,patches,run['scores'],run['angles'])):
        seed_rc=np.floor((seed[1:3]-center_search)*ratio+center_original+.5)-1
        y,x,ncc,boundary=workflow.register_patch(image,patch,*seed_rc)
        if ncc<.5 or boundary:
            raise ValueError(f'Patch registration needs review: {run["label"]}, particle {i+1}, NCC {ncc:.3f}.')
        records.append(dict(particle_id=i+1,score=float(score),x=float(x),y=float(y),
            registration_ncc=ncc,frealign_psi=float(angle[1]),
            frealign_theta=float(angle[2]),frealign_phi=float(angle[3]),
            defocus_u_A=float(angle[8]),defocus_v_A=float(angle[9]),
            astigmatism_angle_deg=float(angle[10]),
            template_image=f'{i+1:06d}@{run["output"]/"templates.mrc"}'))
    return records,image.shape


def verify_rotation(volume,rows,run,root):
    names=('A','AT','DAD','DATD','DA','AD','DAT','ATD')
    p=overlay.parameters(run['path']/'search.par')
    fvolume=overlay.masked_fourier(volume,p)
    reports=[]
    for row in sorted(rows,key=lambda r:-r['score'])[:2]:
        saved=overlay.stack_image(row['template_image'],root)
        if saved.shape!=volume.shape[1:]:
            raise ValueError('Saved template and SP projection dimensions differ.')
        transfer=overlay.ctf(row,saved.shape[0],p)
        scores={name:overlay.correlation(saved,overlay.template_projection(
                    fvolume,overlay.rotation(row,name),transfer)) for name in names}
        reports.append(dict(particle_id=row['particle_id'],**scores))
    if not reports:
        raise ValueError('No templates available to verify rotation conversion.')
    medians={name:float(np.median([r[name] for r in reports])) for name in names}
    ranked=sorted(names,key=lambda n:-medians[n])
    name=ranked[0]
    if medians[name]<.97 or min(r[name] for r in reports)<.94 or medians[name]-medians[ranked[1]]<.02:
        raise ValueError(f'Rotation conversion needs review: {medians}.')
    print(f'{run["label"]}: rotation {name}, median template NCC {medians[name]:.4f}',flush=True)
    return name,dict(convention=name,median_template_NCC=medians[name],checks=reports)


def cif_atom_points(cif):
    # Domain-omission anchors intentionally lack E3 residues, so do not use
    # the surface-coloring reader's full-RNF213 domain validation here.
    import gemmi
    block=gemmi.cif.read_file(str(cif)).sole_block()
    table=block.find('_atom_site.', ['group_PDB','type_symbol','Cartn_x',
                                   'Cartn_y','Cartn_z','pdbx_PDB_model_num'])
    points=[];weights=[]
    for row in table:
        if row[0]!='ATOM' or row[5]!='1':
            continue
        points.append([float(row[j]) for j in (2,3,4)])
        weights.append(float(gemmi.Element(row[1]).atomic_number))
    if not points:
        raise ValueError(f'No model-1 ATOM records in {cif}.')
    points=np.array(points);weights=np.array(weights)
    if not np.isfinite(points).all() or np.any(weights<=0):
        raise ValueError('Invalid atom coordinates or element weights.')
    return points,weights


def cif_origin(volume,cif,apix):
    points,weights=cif_atom_points(cif)
    center=np.average(points,axis=0,weights=weights)
    # Symmetric padding preserves the physical location of the grid origin.
    if volume.shape[0]%4:
        volume=np.pad(volume,1,constant_values=overlay.mode_value(volume))
    _,info=overlay.register_atoms(volume,points,weights,apix)
    scale=info['grid_scale_correction']
    translation=np.array(info['translation_original_px_xyz'])
    origin=center-apix*translation/scale
    info.update(CIF=str(cif),atomic_number_weighted_COM_CIF_xyz_A=center.tolist(),
                SP_origin_in_CIF_xyz_A=origin.tolist())
    return origin,center,scale,translation,info


def assignments(points,targets,radius):
    if not len(points) or not len(targets):
        return {}
    distances=np.linalg.norm(np.array(points)[:,None,:]-np.array(targets)[None,:,:],axis=2)
    penalty=(len(points)+len(targets)+1)*radius
    cost=np.full((len(points),len(targets)+len(points)),penalty)
    cost[:,:len(targets)]=np.where(distances<=radius,distances,penalty*3)
    ii,jj=linear_sum_assignment(cost)
    return {int(i):(int(j),float(distances[i,j])) for i,j in zip(ii,jj)
            if j<len(targets) and distances[i,j]<=radius}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fragment','--partial-run',dest='fragment',default='fragment_200kDa_pilot')
    p.add_argument('--snr-threshold',type=float,default=8.)
    p.add_argument('--match-radius',type=float,default=20.)
    args=p.parse_args()
    if not np.isfinite([args.snr_threshold,args.match_radius]).all() or args.match_radius<=0:
        p.error('Use a finite threshold and positive match radius.')
    root=Path.cwd().resolve();fragment=Path(args.fragment).resolve()
    manifest=json.loads((fragment/'manifest.json').read_text())
    full=Path(manifest['source_full_template_section']).resolve()
    fullrun=compare.load_run(full,'Full template',root)
    kind='anchor' if 'domain-omission' in manifest.get('template_type','') else 'fragment'
    partial_label='E3/CTD-omitted anchor' if kind=='anchor' else 'Fragment template'
    fragrun=compare.load_run(fragment,partial_label,root)
    images=[compare.host_path(r['params']['imageFile'],root) for r in (fullrun,fragrun)]
    if compare.digest(images[0])!=compare.digest(images[1]) or compare.digest(images[1])!=manifest['source_image_sha256']:
        raise ValueError('Different images cannot be compared as a same-tile recovery test.')
    for key in ('angle_inc','psi_inc','aPerPix','aPerPix_search','thickness','df_inc',
                'arbThr','optThr','inc_degrees'):
        if fullrun['params'].get(key)!=fragrun['params'].get(key):
            raise ValueError(f'Search parameter differs: {key}.')
    source_cif=Path(manifest['source_CIF'])
    if compare.digest(source_cif)!=manifest['source_CIF_sha256']:
        raise ValueError('Source CIF changed since fragment preparation.')
    apix=float(fullrun['params']['aPerPix'])
    if not np.isclose(apix,float(fragrun['params']['aPerPix'])):
        raise ValueError('Pixel sizes differ.')
    fullrows,shape=register_positions(fullrun,root)
    fragrows,fragshape=register_positions(fragrun,root)
    if fragshape!=shape:
        raise ValueError('Tile dimensions differ.')
    fullvol=load_sp(sp_path(fullrun,root),apix)
    fullconv,fullcheck=verify_rotation(fullvol,fullrows,fullrun,root)
    fullorigin,_,_,_,fullinfo=cif_origin(fullvol,source_cif,apix)
    del fullvol
    fragvol=load_sp(sp_path(fragrun,root),apix)
    fragconv,fragcheck=verify_rotation(fragvol,fragrows,fragrun,root)
    fragcif=fragment/manifest['fragment_CIF']
    _,fragcenter,fragscale,fragtranslation,fraginfo=cif_origin(fragvol,fragcif,apix)
    del fragvol
    # The FULL SP origin expressed in FRAGMENT SP pixel coordinates.
    # Rotate this displacement with the fragment pose, then add to its hit.
    offset=(fullorigin-fragcenter)/apix*fragscale+fragtranslation
    with (full.parent/'tile_offsets.csv').open(newline='') as f:
        offsets=list(csv.DictReader(f))
    tile=next(o for o in offsets if o['tile']==full.name)
    x0,y0=(int(tile[k]) for k in ('x0_zero_based','y0_zero_based'))
    accepted_full=[r for r in fullrows if r['score']>=args.snr_threshold]
    accepted_frag=[r for r in fragrows if r['score']>=args.snr_threshold]
    for row in fragrows:
        shift=overlay.rotation(row,fragconv)@offset
        row.update(parent_x=row['x']+shift[0],parent_y=row['y']+shift[1],
                   corrected_global_x=row['x']+shift[0]+x0,
                   corrected_global_y=row['y']+shift[1]+y0)
    matches=assignments([(r['parent_x'],r['parent_y']) for r in accepted_frag],
                        [(r['x'],r['y']) for r in accepted_full],args.match_radius)
    reference_path=full.parent/'review/merged_v3/reference_particles.csv'
    if not reference_path.exists():
        reference_path=full.parent/'review/merged/reference_particles.csv'
    refs=[]
    if reference_path.exists():
        with reference_path.open(newline='') as f:
            refs=[r for r in csv.DictReader(f) if x0-.5<=float(r['x'])<x0+shape[1]-.5
                  and y0-.5<=float(r['y'])<y0+shape[0]-.5]
    refmatches=assignments([(r['corrected_global_x'],r['corrected_global_y']) for r in accepted_frag],
                           [(float(r['x']),float(r['y'])) for r in refs],args.match_radius)
    results=[]
    for i,row in enumerate(accepted_frag):
        match=matches.get(i)
        target=accepted_full[match[0]] if match else None
        angle=''
        if target:
            relative=overlay.rotation(row,fragconv)@overlay.rotation(target,fullconv).T
            angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        ref=refmatches.get(i)
        results.append(dict(fragment_particle_id=row['particle_id'],fragment_SNR=row['score'],
            full_particle_id=target['particle_id'] if target else '',full_SNR=target['score'] if target else '',
            corrected_distance_pixels=match[1] if match else '',rotation_difference_degrees=angle,
            fragment_center_x=row['x'],fragment_center_y=row['y'],
            corrected_parent_center_x=row['parent_x'],corrected_parent_center_y=row['parent_y'],
            corrected_global_x=row['corrected_global_x'],corrected_global_y=row['corrected_global_y'],
            reference_id=refs[ref[0]]['reference_id'] if ref else '',
            reference_distance_pixels=ref[1] if ref else ''))
    out=fragment/'review/position_check';out.mkdir(parents=True,exist_ok=True)
    fields=list(results[0]) if results else ['fragment_particle_id','fragment_SNR','full_particle_id',
                                          'corrected_distance_pixels','rotation_difference_degrees']
    csv_write(out/(kind+'_full_matches.csv'),results,fields)
    csv_write(out/('registered_'+kind+'_candidates.csv'),fragrows,list(fragrows[0]))
    csv_write(out/'registered_full_candidates.csv',fullrows,list(fullrows[0]))
    (out/'center_registration.json').write_text(json.dumps(dict(
        full=fullinfo,fragment=fraginfo,full_rotation_check=fullcheck,fragment_rotation_check=fragcheck,
        full_SP_origin_in_fragment_SP_xyz_pixels=offset.tolist(),
        precision='Coarse 4-pixel CIF/SP registration with subpixel interpolation; approximate centers',
        cutoff=args.snr_threshold,match_radius_pixels=args.match_radius),indent=2)+'\n')
    lines=[f'Accepted full-template candidates: {len(accepted_full)}',
           f'Accepted {kind} candidates: {len(accepted_frag)}',
           f'One-to-one {kind}/full positional matches: {len(matches)}',
           f'{kind.capitalize()} reference-position matches: {len(refmatches)} / {len(refs)} reference picks in tile footprint'
           if reference_path.exists() else 'Reference-position check unavailable: no registered reference CSV found.']
    for row in results:
        if row['full_particle_id']:
            lines.append(f'{kind.capitalize()} {row["fragment_particle_id"]} -> full {row["full_particle_id"]}: '
                         f'{row["corrected_distance_pixels"]:.2f} px; rotation difference '
                         f'{row["rotation_difference_degrees"]:.2f} degrees')
    lines.extend(['Template poses are internally checked, not independently biologically validated.',
                  'The full template is a comparison set, not an exhaustive ground truth.',
                  'Reference picks are selected; unmatched candidates are not automatically false positives.'])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print(f'Saved: {out}')


if __name__=='__main__':
    main()
