#!/usr/bin/env python3
"""Retain a completed image prefix; archive excluded inputs/results reversibly.

Run from the RNF213 project. Default is a preview; --apply changes the active
CSV/STAR/JSON manifests and moves excluded images into a dated archive.
Completed search directories, configurations and data remain at their old paths.
No searches, particle filtering or reconstruction are performed here.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def atomic_bytes(path, content):
    stat = path.stat()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.st_mode & 0o777)
        if os.geteuid() == 0:
            os.chown(temporary, stat.st_uid, stat.st_gid)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_plan(root, batch, limit, workflow, planner):
    paths = [batch/'selected_micrographs.csv', batch/'selected_micrographs_ctf.star',
             batch/'batch_plan.json', batch/'download_report.json',
             batch/'searches/jobs.csv', batch/'searches/batch_search.json']
    require(all(p.is_file() and not p.is_symlink() for p in paths),
            'Missing or symlinked batch metadata; no cleanup performed.')
    originals = {p: p.read_bytes() for p in paths}
    selection = workflow.read_csv(paths[0])
    jobs = workflow.read_csv(batch/'searches/jobs.csv')
    details = json.loads(originals[batch/'searches/batch_search.json'])
    require(Path(details['project']).resolve() == root and
            Path(details['batch']).resolve() == batch, 'Batch/project provenance mismatch.')
    require(selection and jobs, 'Empty selection or job list.')
    names = [r['batch_image'] for r in selection]
    require(len(set(names)) == len(names) and all(re.fullmatch(r'image_\d{4}', n) for n in names),
            'Invalid or duplicate image identifiers.')
    require({j['batch_image'] for j in jobs} == set(names), 'Selection and prepared jobs differ.')
    require(len({r['basename'] for r in selection}) == len(selection), 'Duplicate micrograph basenames.')
    require(all(Path(r['basename']).name == r['basename'] and r['basename'].endswith('.mrc')
                for r in selection), 'Unsafe micrograph basename.')
    kept = {n for n in names if int(n.split('_')[1]) <= limit}
    require(kept == {f'image_{i:04d}' for i in range(1, limit+1)},
            'Requested completed image prefix is not present.')
    dropped = set(names) - kept
    retained_jobs, excluded, moves = [], [], []
    for row in selection:
        name = row['batch_image']
        directory = batch/'searches'/name
        image = batch/'micrographs'/row['basename']
        require(directory.is_dir() and not directory.is_symlink(), f'Invalid image directory: {directory}')
        require(image.is_file() and not image.is_symlink(), f'Invalid source image: {image}')
        image_jobs = [j for j in jobs if j['batch_image'] == name]
        offsets = workflow.read_csv(directory/'tile_offsets.csv')
        require(len(image_jobs) == len(offsets) == 54 and
                {j['tile'] for j in image_jobs} == {r['tile'] for r in offsets},
                f'Expected 54 distinct prepared tiles: {name}')
        completed = 0
        for job in image_jobs:
            require(re.fullmatch(r'tile_\d{4}', job['tile']), 'Unsafe tile name.')
            tile = workflow.project_path(root, job['directory'])
            require(tile == directory/job['tile'] and not tile.is_symlink(),
                    f'Unexpected tile path: {tile}')
            done = workflow.complete(tile)
            completed += int(done)
            if name in kept:
                require(done, f'{name}/{job["tile"]} is incomplete; nothing changed.')
                require(Path(job['tile_file']).name == job['tile_file'], 'Unsafe tile input filename.')
                require(workflow.digest(tile/'search.par') == job['configuration_sha256'] and
                        workflow.digest(tile/job['tile_file']) == job['tile_sha256'],
                        f'Prepared input changed: {tile}')
        if name in kept:
            meta = json.loads((directory/'micrograph_manifest.json').read_text())
            require(Path(meta['source_micrograph']).resolve() == image and
                    workflow.digest(image) == meta['source_micrograph_sha256'],
                    f'Source micrograph changed: {name}')
            retained_jobs.extend(image_jobs)
        else:
            excluded.append(dict(batch_image=name, completed_tiles=completed, planned_tiles=len(image_jobs)))
            moves.extend([directory, image])
            part = image.with_name(image.name+'.part')
            if part.exists():
                require(part.is_file() and not part.is_symlink(), f'Unsafe partial image: {part}')
                moves.append(part)
    # Preserve every optics value; filter only micrograph rows in the selected STAR.
    loops = planner.read_loops(batch/'selected_micrographs_ctf.star')
    micro_loops = [i for i, (_, labels, _) in enumerate(loops) if '_rlnMicrographName' in labels]
    require(len(micro_loops) == 1, 'Expected one micrograph STAR loop.')
    i = micro_loops[0]
    block, labels, star_rows = loops[i]
    basenames = {r['basename'] for r in selection if r['batch_image'] in kept}
    filtered = [r for r in star_rows if Path(r['_rlnMicrographName']).name in basenames]
    require(len(filtered) == len(kept) and
            {Path(r['_rlnMicrographName']).name for r in filtered} == basenames,
            'Selected STAR does not uniquely identify all retained images.')
    loops[i] = (block, labels, filtered)
    # Render the STAR to a temporary scratch file without touching original data.
    with tempfile.TemporaryDirectory(prefix='rnf213-finalize-') as temp:
        star = Path(temp)/'selected.star'
        planner.write_loops(star, loops)
        star_content = star.read_bytes()
    plan = json.loads(originals[batch/'batch_plan.json'])
    download = json.loads(originals[batch/'download_report.json'])
    require({r['batch_image'] for r in plan['images']} == set(names) and
            {r['batch_image'] for r in download['images']} == set(names),
            'Plan/download metadata differs from active selection.')
    plan['images'] = [r for r in plan['images'] if r['batch_image'] in kept]
    plan['selected_images'] = len(kept)
    plan['status'] = 'Completed image subset; excluded inputs/results archived'
    download['images'] = [r for r in download['images'] if r['batch_image'] in kept]
    details.update(total_images=len(kept), total_tiles=len(retained_jobs))
    updated = {
        batch/'selected_micrographs.csv': csv_bytes([r for r in selection if r['batch_image'] in kept]),
        batch/'selected_micrographs_ctf.star': star_content,
        batch/'batch_plan.json': json_bytes(plan),
        batch/'download_report.json': json_bytes(download),
        batch/'searches/jobs.csv': csv_bytes(retained_jobs),
        batch/'searches/batch_search.json': json_bytes(details),
    }
    return dict(originals=originals, updated=updated, moves=moves, excluded=excluded,
                kept=sorted(kept), dropped=sorted(dropped), retained_jobs=len(retained_jobs))


def apply_plan(batch, plan):
    archive = batch/'excluded_archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive.mkdir(parents=True)
    moved, rewritten = [], []
    for path, content in plan['originals'].items():
        backup = archive/'original_metadata'/path.relative_to(batch)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(content)
    report = dict(status='in_progress', retained_images=plan['kept'], excluded_images=plan['excluded'],
                  active_tiles=plan['retained_jobs'], archive=str(archive),
                  moves=[str(p.relative_to(batch)) for p in plan['moves']],
                  original_metadata_SHA256={str(p.relative_to(batch)): hashlib.sha256(b).hexdigest()
                                            for p,b in plan['originals'].items()})
    report_path = archive/'cleanup_report.json'
    report_path.write_bytes(json_bytes(report))
    try:
        for source in plan['moves']:
            destination = archive/'excluded_data'/source.relative_to(batch)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append((source, destination))
        for path, content in plan['updated'].items():
            atomic_bytes(path, content)
            rewritten.append(path)
    except BaseException:
        for path in reversed(rewritten):
            atomic_bytes(path, plan['originals'][path])
        for source, destination in reversed(moved):
            destination.rename(source)
        report['status'] = 'rolled_back'
        report_path.write_bytes(json_bytes(report))
        raise
    report['status'] = 'completed'
    report['active_metadata_SHA256'] = {str(p.relative_to(batch)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in plan['updated']}
    report_path.write_bytes(json_bytes(report))
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', default='batch_noE3_30')
    p.add_argument('--keep-through', type=int, default=21)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    root = Path.cwd().resolve()
    batch_input = Path(a.batch).absolute()
    require(not batch_input.is_symlink(), 'Batch path must not be a symlink.')
    batch = batch_input.resolve()
    require(batch.parent == root and a.keep_through > 0, 'Run from the project root with a direct batch subdirectory.')
    sys.path.insert(0, str(root))
    import rnf213_batch_workflow as workflow
    import rnf213_batch_plan as planner
    with (batch/'searches/.runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Batch runner is still active. Stop it with Ctrl+C in its terminal, then rerun.')
        if a.apply:
            result = subprocess.run(['/usr/bin/docker', 'ps', '--format', '{{.Names}}'],
                                    capture_output=True, text=True, check=True)
            prefix = 'rnf213-anchor-'+hashlib.sha256(str(batch).encode()).hexdigest()[:8]+'-'
            require(not any(n.startswith(prefix) for n in result.stdout.splitlines()),
                    'A container for this batch is still running; no cleanup performed.')
        print('Checking completed search files, configurations and source-image hashes...', flush=True)
        plan = build_plan(root, batch, a.keep_through, workflow, planner)
        print(f'Retain {len(plan["kept"])} complete images / {plan["retained_jobs"]} tiles.')
        for row in plan['excluded']:
            print(f'Archive {row["batch_image"]}: {row["completed_tiles"]}/{row["planned_tiles"]} complete tiles.')
        if not plan['dropped']:
            print('Active batch already contains only the requested complete images. Nothing changed.')
        elif a.apply:
            archive = apply_plan(batch, plan)
            print(f'Archived excluded inputs/results: {archive}')
            print('Active CSV, STAR, download, planning and job metadata updated.')
            print(f'Batch directory name remains {batch.name} to preserve existing search paths.')
            print('Ready for batch review. Archive is reversible and does not free disk space.')
        else:
            print('Preview only. Add --apply to archive exclusions and update active metadata.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Cleanup stopped: {exc}', file=sys.stderr)
        sys.exit(1)
