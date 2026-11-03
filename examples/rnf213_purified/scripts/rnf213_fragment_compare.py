#!/usr/bin/env python3
"""Compare a completed spherical-fragment pilot with its exact full-template tile.

Plots retained-search survival tails, final-score counts, and experimental/template
pairs. This does not infer positional recovery: independently centered fragment
and full models require a validated, orientation-dependent center correction.
Run from the RNF213 project directory; no Docker or GPU search is launched.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import mrcfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def params(path):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split('#',1)[0].split(maxsplit=1)
        if len(fields)==2:
            result[fields[0]] = fields[1]
    return result


def host_path(value, root):
    if value.startswith('/work/'):
        return root/value[6:]
    path = Path(value)
    return path if path.is_absolute() else root/path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def scores(folder, log):
    path=folder/'particles.txt'
    if path.exists():
        lines=[line for line in path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
        data=np.loadtxt(lines,ndmin=2) if lines else np.empty((0,2))
    elif 'nothing to optimize' in log and 'starting optimization on' not in log:
        data=np.empty((0,2))
    else:
        raise FileNotFoundError(path)
    if data.shape[1]!=2 or not np.isfinite(data).all():
        raise ValueError(f'Invalid final scores in {path}.')
    if not np.array_equal(data[:,0],np.arange(1,len(data)+1)):
        raise ValueError('Particle IDs must match the one-worker stack order.')
    return data


def load_run(path, label, root):
    log=(path/'run.log').read_text()
    if 'Done combining search output' not in log:
        raise ValueError(f'Search is incomplete: {path}.')
    p=params(path/'search.par')
    if int(p['nCores'])!=1:
        raise ValueError('This comparison expects the established single-worker searches.')
    output=host_path(p['outputDir'],root)
    data=scores(output,log)
    sh=np.loadtxt(output/'search_SH.txt',ndmin=2)
    if sh.shape[1]!=3 or not np.isfinite(sh).all() or np.any(sh[:,1:]<0):
        raise ValueError(f'Invalid survival table: {output}.')
    if np.any(np.diff(sh[:,0])<0) or np.any(np.diff(sh[:,2])>1e-6):
        raise ValueError('Survival thresholds must increase and counts must not increase.')
    if len(data):
        angle_lines=[line for line in (output/'particles.par').read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith(('C','#'))]
        angles=np.loadtxt(angle_lines,ndmin=2)
        if len(angles)!=len(data) or not np.array_equal(angles[:,0],data[:,0]):
            raise ValueError('Angles and final score stack indexing disagree.')
    else:
        angles=np.empty((0,4))
    return dict(path=path,label=label,params=p,output=output,scores=data[:,1],
                angles=angles,sh=sh)


def final_counts(values, thresholds):
    return np.array([np.count_nonzero(values>=x) for x in thresholds],dtype=int)


def save_figure(fig,folder,name):
    for ext in ('png','pdf'):
        fig.savefig(folder/f'{name}.{ext}',dpi=250)
    plt.close(fig)


def comparison_plots(runs,cutoff,out):
    colors=['#2878B5','#A465B7']
    fig,ax=plt.subplots(figsize=(7.4,5),layout='constrained')
    for run,color in zip(runs,colors):
        x,gaussian,measured=run['sh'].T
        valid=gaussian>0
        ax.plot(x[valid],gaussian[valid],color=color,ls='--',lw=1.2,
                label=f"{run['label']}: Gaussian expectation")
        valid=measured>0
        ax.step(x[valid],measured[valid],where='post',color=color,lw=1.5,
                label=f"{run['label']}: retained CC tail")
    ax.axvline(cutoff,color='#666666',ls=':',lw=1)
    ax.set(yscale='log',xlabel='Flat-fielded search SNR threshold',
           ylabel='Surviving CC evaluations',
           title='Same image tile · full versus partial-template search\nSearch CC tails before refinement')
    ax.legend(frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.15)
    save_figure(fig,out,'search_survival_comparison')

    all_scores=np.concatenate([r['scores'] for r in runs])
    low=min(6.,float(all_scores.min())-.2) if len(all_scores) else 6.
    high=max(10.,float(all_scores.max())+.3) if len(all_scores) else 10.
    thresholds=np.unique(np.r_[np.linspace(low,high,800),all_scores,cutoff])
    fig,ax=plt.subplots(figsize=(7.4,4.8),layout='constrained')
    table=[]
    for run,color in zip(runs,colors):
        values=run['scores']
        counts=final_counts(values,thresholds)
        # For >= cutoff counts, the drop is immediately AFTER each stored score.
        ax.step(thresholds,counts,where='pre',color=color,lw=1.8,
                label=f"{run['label']} · {np.count_nonzero(values>=cutoff)} at SNR ≥ {cutoff:g}")
        for value,count in zip(thresholds,counts):
            table.append(dict(template=run['label'],final_SNR_cutoff=float(value),
                              detected_particle_candidates=int(count)))
    ax.axvline(cutoff,color='#555555',ls=':',lw=1.3)
    ax.set(xlabel='Final optimized SNR cutoff',ylabel='Detected particle candidates at or above cutoff',
           title='Cumulative final-score counts · same tile',ylim=(-.15,None))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=.15)
    save_figure(fig,out,'final_score_counts')
    with (out/'final_score_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)


def read_stack(path):
    with mrcfile.open(path) as m:
        data=np.array(m.data,dtype=np.float32,copy=True)
    return data[None] if data.ndim==2 else data


def particle_pairs(run,cutoff,out,max_show=8):
    values=run['scores']
    if not len(values):
        return
    patches=read_stack(run['output']/'particles.mrc')
    templates=read_stack(run['output']/'templates.mrc')
    if len(patches)!=len(values) or len(templates)!=len(values):
        raise ValueError('Particle/template stacks do not match score rows.')
    indices=np.argsort(-values)[:max_show]
    fig,axes=plt.subplots(len(indices),2,figsize=(7,3.1*len(indices)),
                          squeeze=False,layout='constrained')
    apix=float(run['params']['aPerPix'])
    for i,j in enumerate(indices):
        patch,template=patches[j],templates[j]
        for ax,image,title in zip(axes[i],(patch,template),('Experimental patch','Saved SMAP template')):
            lo,hi=np.percentile(image,[1,99])
            h,w=image.shape
            ax.imshow(image,cmap='gray',origin='lower',vmin=lo,vmax=hi,
                      extent=(-w*apix/20,w*apix/20,-h*apix/20,h*apix/20))
            ax.set_title(title,fontsize=10)
            ax.set_xlabel('nm')
            ax.set_ylabel('nm')
        status='passes' if values[j]>=cutoff else 'below cutoff'
        axes[i,0].set_title(f'Particle {j+1} · final SNR {values[j]:.3f} · {status}',fontsize=10)
    fig.suptitle(run['label']+' · experimental/template pairs\n'
                 'Contrast adjusted separately per panel; native physical scale')
    name=run.get('kind','fragment' if run['label'].startswith('Fragment') else 'full')+'_particle_pairs'
    save_figure(fig,out,name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fragment','--partial-run',dest='fragment',default='fragment_200kDa_pilot',
                   help='Completed spherical-fragment or domain-omission anchor folder.')
    p.add_argument('--full-tile',help='Defaults to the baseline tile recorded in manifest.json.')
    p.add_argument('--snr-threshold',type=float,default=8.)
    p.add_argument('--out',help='Defaults to fragment directory/review.')
    args=p.parse_args()
    if not np.isfinite(args.snr_threshold):
        p.error('SNR threshold must be finite.')
    root=Path.cwd().resolve()
    fragment=Path(args.fragment).resolve()
    manifest=json.loads((fragment/'manifest.json').read_text())
    full=Path(args.full_tile or manifest['source_full_template_section']).resolve()
    mass=float(manifest['approximate_fragment_mass_kDa'])
    kind='anchor' if 'domain-omission' in manifest.get('template_type','') else 'fragment'
    if kind=='anchor':
        omitted='E3/CTD' if manifest.get('omitted_CTD_author_residues') else 'E3'
        label=f'Anchor without {omitted} · {mass:.0f} kDa · 3.8°'
    else:
        label=f'Fragment {mass:.0f} kDa · 3.8°'
    runs=[load_run(full,'Full template · 3.8°',root),load_run(fragment,label,root)]
    runs[0]['kind']='full';runs[1]['kind']=kind
    images=[host_path(r['params']['imageFile'],root) for r in runs]
    hashes=[digest(path) for path in images]
    if hashes[0]!=hashes[1] or hashes[1]!=manifest['source_image_sha256']:
        raise ValueError('The runs do not use the exact same source image bytes.')
    checked=[]
    for key in ('angle_inc','psi_inc','aPerPix','aPerPix_search','thickness',
                'df_inc','arbThr','optThr','inc_degrees'):
        values=[r['params'].get(key) for r in runs]
        if values[0]!=values[1]:
            raise ValueError(f'Search parameter differs: {key}: {values}.')
        if values[0] is not None:
            checked.append(key)
    if not all(np.isclose(float(r['params'][key]),3.8) for r in runs
               for key in ('angle_inc','psi_inc')):
        raise ValueError('Both runs must use 3.8° global angular spacing.')
    out=Path(args.out).resolve() if args.out else fragment/'review'
    out.mkdir(parents=True,exist_ok=True)
    comparison_plots(runs,args.snr_threshold,out)
    records=[]
    lines=['Same tile, identical source image bytes; both searches completed.',
           f'Final optimized SNR criterion: >= {args.snr_threshold:g}']
    for run in runs:
        values=run['scores']
        accepted=int(np.count_nonzero(values>=args.snr_threshold))
        lines.append(f"{run['label']}: {len(values)} refined candidates; {accepted} pass final cutoff.")
        lines.append('  Final exported scores: '+', '.join(f'{v:.3f}' for v in values))
        for i,value in enumerate(values):
            a=run['angles'][i]
            records.append(dict(template=run['label'],particle_id=i+1,final_SNR=float(value),
                                passes_cutoff=bool(value>=args.snr_threshold),
                                frealign_psi=float(a[1]),frealign_theta=float(a[2]),frealign_phi=float(a[3])))
        particle_pairs(run,args.snr_threshold,out)
    fields=['template','particle_id','final_SNR','passes_cutoff',
            'frealign_psi','frealign_theta','frealign_phi']
    with (out/'particle_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    lines.extend(['Search survival counts are retained CC evaluations, not particles.',
        'A single-tile partial-template pilot cannot support a cumulative comparison across 41 sections.',
        'Positional/reference recovery is not measured by these counts.',
        'Partial and full template centers must be registered and transformed before positional matching.',
        'Different template sizes and refined scores can have different noise behavior.',
        'Source tile was chosen for strong full-template hits; this is an exploratory sensitivity test.'])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    (out/'comparison_settings.json').write_text(json.dumps(dict(
        full_tile=str(full),fragment_run=str(fragment),source_image_sha256=hashes[0],
        cutoff=args.snr_threshold,checked_equal_parameters=checked,
        coordinate_matches_performed=False,
        fragment_COM_offset_CIF_xyz_A=manifest.get('fragment_COM_minus_parent_RNF213_COM_A'),
        center_warning='Atomic mass COM difference is recorded only; it is not a validated SP-grid center correction'),
        indent=2)+'\n')
    print('\n'.join(lines[:6]))
    print(f'Saved comparison: {out}')
    print('No new searches launched. Position/reference matching awaits model-center registration.')


if __name__=='__main__':
    main()
