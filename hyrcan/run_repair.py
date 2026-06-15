import hyrcan as hy
import numpy as np
import math, csv, io, contextlib, os

INFILE  = 'pilot_50.csv'                  # source CSV with FOS results to diagnose
OUTFILE = 'at_boundary_diagnostic.csv'    # diagnostic output CSV
METHOD  = 'Spencer'                       # slope stability method passed to the solver

# Geometry rescaling factors to retry for cases flagged At_Boundary=True
GEOMETRY_VARIANTS = {
    'original':  {'ext_factor': None,  'depth_factor': None},
    'ext_2x':    {'ext_factor': 2.0,   'depth_factor': None},
    'depth_2x':  {'ext_factor': None,  'depth_factor': 2.0},
    'both_1_5x':   {'ext_factor': 1.5,   'depth_factor': 1.5},
    'both_2x':   {'ext_factor': 2.0,   'depth_factor': 2.0},
    'both_3x':   {'ext_factor': 3.0,   'depth_factor': 3.0},
}

# Columns written to the diagnostic output CSV
DIAG_COLS = [
    'case_idx', 'angle', 'phi', 'c', 'gamma', 'H', 'fos_original_csv',
    'variant', 'ext_factor', 'depth_factor',
    'fos', 'delta_fos', 'cx', 'cy', 'r',
    'slope_length', 'extension', 'depth',
    'at_boundary', 'boundary_flags', 'solver_output', 'error'
]


# ── Geometry ──────────────────────────────────────────────────────────────────
def model_dims(H, angle, ext_factor=None, depth_factor=None):
    """Derive slope ratio, slope length, extension and excavation depth from H and angle."""
    slope        = math.tan(angle * math.pi / 180)
    slope_length = H / slope
    extension    = max(1.5 * H, slope_length * 0.5)
    depth        = max(H, 0.5 * slope_length)
    if ext_factor:
        extension *= ext_factor
    if depth_factor:
        depth *= depth_factor
    return slope, slope_length, extension, depth


def build_cmd(H, angle, phi, c, gamma, ext_factor=None, depth_factor=None):
    """Build the solver command script for the given geometry/material parameters."""
    slope, slope_length, extension, depth = model_dims(
        H, angle, ext_factor, depth_factor)
    cmd = f"""
    newmodel()
    set('failureDir','r2l')
    extboundary(-{depth},-{depth},-{depth},0,0,0,{slope_length},{H},{slope_length + extension},{H},{slope_length + extension},-{depth},-{depth},-{depth})
    definemat('ground','matID',1,'matName','Soil','uw',{gamma},'cohesion',{c},'friction',{phi})
    assignsoilmat('matid',1,'atpoint',{slope_length/2},{H/2})
    definelimits('limit',-{depth},{H/3/slope},'limit2',{2*H/3/slope},{slope_length + extension})
    set('Method','{METHOD}','on')
    """
    return cmd, slope_length, extension, depth


# ── QC ───────────────────────────────────────────────────────────────────────
def check_boundary(cx, cy, r, H, angle, ext_factor=None, depth_factor=None):
    """Check whether the critical slip surface touches the model's left, right or bottom boundary."""
    _, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    tol   = 0.01 * H
    flags = []

    def intersect(y):
        if abs(cy - y) > r:
            return None
        dx = math.sqrt(max(r**2 - (cy - y)**2, 0))
        return (cx - dx, cx + dx)

    gi = intersect(0)
    if gi and abs(gi[0] - (-depth)) < tol:
        flags.append(f'left_lower: x={gi[0]:.3f} vs -{depth:.3f}')

    ti = intersect(H)
    if ti and abs(ti[1] - (slope_length + extension)) < tol:
        flags.append(f'right_upper: x={ti[1]:.3f} vs {slope_length+extension:.3f}')

    if cy - r < -depth:
        flags.append(f'below_domain: cy-r={cy-r:.3f} vs -{depth:.3f}')

    return bool(flags), flags


# ── CSV reading without pandas ─────────────────────────────────────────────────
def read_csv(path):
    """Read a CSV file into a list of dicts (one per row)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val, default=None):
    """Convert val to float, returning default if conversion fails."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Run variant ────────────────────────────────────────────────────────────────
def run_variant(angle, phi, c, gamma, H, ext_factor, depth_factor, variant_name):
    """Run the solver with one geometry variant and return its FOS, surface and boundary results."""
    res = {
        'variant':        variant_name,
        'ext_factor':     ext_factor or 'auto',
        'depth_factor':   depth_factor or 'auto',
        'fos':            None, 'cx': None, 'cy': None, 'r': None,
        'slope_length':   None, 'extension': None, 'depth': None,
        'at_boundary':    None, 'boundary_flags': '',
        'solver_output':  '', 'error': '',
    }
    try:
        cmd, slope_length, extension, depth = build_cmd(
            H, angle, phi, c, gamma, ext_factor, depth_factor)
        res.update({
            'slope_length': round(slope_length, 3),
            'extension':    round(extension, 3),
            'depth':        round(depth, 3),
        })

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hy.command(cmd)
            hy.command("compute('silence')")
            fos    = hy.min_fos(METHOD)
            center = hy.surf_center_min_fos(METHOD)
            cx, cy = center[0], center[1]
            r      = hy.surf_radius_min_fos(METHOD)

        res['solver_output'] = buf.getvalue().strip().replace('\n', ' | ')
        res['fos'] = round(fos, 5)
        res['cx']  = round(cx, 4)
        res['cy']  = round(cy, 4)
        res['r']   = round(r, 4)

        at_b, flags = check_boundary(cx, cy, r, H, angle, ext_factor, depth_factor)
        res['at_boundary']    = at_b
        res['boundary_flags'] = '; '.join(flags)

    except Exception as e:
        import traceback
        res['error'] = f'{type(e).__name__}: {e} | {traceback.format_exc()}'

    return res


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Rerun every At_Boundary=True case from INFILE with each geometry variant and write a diagnostic CSV."""
    all_rows = read_csv(INFILE)

    # filter rows with At_Boundary=True and a valid FOS
    bad = []
    for i, row in enumerate(all_rows):
        fos_val = to_float(row.get('FOS'))
        at_b    = str(row.get('At_Boundary', '')).strip().lower() == 'true'
        if fos_val is not None and at_b:
            bad.append((i, row, fos_val))

    if not bad:
        print('No cases with At_Boundary=True. Nothing to diagnose.')
        return

    print(f'{len(bad)} cases with At_Boundary=True — rerunning with '
          f'{len(GEOMETRY_VARIANTS)} geometry variants.\n')

    with open(OUTFILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=DIAG_COLS)
        writer.writeheader()

        for orig_idx, row, fos_csv in bad:
            angle = to_float(row['angle']); phi   = to_float(row['phi'])
            c     = to_float(row['c']);     gamma = to_float(row['gamma'])
            H     = to_float(row['H'])

            print(f'─── Case #{orig_idx}: angle={angle:.1f}° φ={phi:.1f}° '
                  f'c={c:.1f} γ={gamma:.1f} H={H:.1f}m  FOS_csv={fos_csv:.3f}')

            for vname, vp in GEOMETRY_VARIANTS.items():
                res = run_variant(angle, phi, c, gamma, H,
                                  vp['ext_factor'], vp['depth_factor'], vname)

                delta = (round(res['fos'] - fos_csv, 5)
                         if res['fos'] is not None else '')

                if res['error']:
                    status = '✗ ERROR'
                elif res['at_boundary']:
                    status = '✗ still_boundary'
                else:
                    status = '✓ RESOLVED'

                print(f'  [{vname:12s}] FOS={str(res["fos"]):>9}  '
                      f'ΔFOS={str(delta):>9}  '
                      f'at_boundary={res["at_boundary"]}  {status}')
                if res['boundary_flags']:
                    print(f'               flags: {res["boundary_flags"]}')
                if res['solver_output']:
                    print(f'               solver: {res["solver_output"][:120]}')
                if res['error']:
                    print(f'               error: {res["error"][:200]}')

                writer.writerow({
                    'case_idx':         orig_idx,
                    'angle':            angle, 'phi': phi, 'c': c,
                    'gamma':            gamma, 'H': H,
                    'fos_original_csv': fos_csv,
                    'delta_fos':        delta,
                    **{k: res[k] for k in [
                        'variant', 'ext_factor', 'depth_factor',
                        'fos', 'cx', 'cy', 'r',
                        'slope_length', 'extension', 'depth',
                        'at_boundary', 'boundary_flags',
                        'solver_output', 'error'
                    ]},
                })

            print()

    # ── Summary from written CSV (no pandas) ──────────────────────────────────
    diag_rows = read_csv(OUTFILE)
    print('=' * 60)
    print('SUMMARY — resolved cases per variant:')
    print(f'  {"variant":<14} {"total":>6} {"resolved":>9} {"ΔFOS_mean":>10} {"ΔFOS_max":>10}')
    for vname in GEOMETRY_VARIANTS:
        vrows = [r for r in diag_rows if r['variant'] == vname and not r['error']]
        if not vrows:
            continue
        n_tot = len(vrows)
        n_res = sum(1 for r in vrows if str(r['at_boundary']).lower() == 'false')
        deltas = [abs(to_float(r['delta_fos'], 0)) for r in vrows if r['delta_fos'] != '']
        d_mean = sum(deltas) / len(deltas) if deltas else 0
        d_max  = max(deltas) if deltas else 0
        print(f'  {vname:<14} {n_tot:>6} {n_res:>9} {d_mean:>10.4f} {d_max:>10.4f}')

    print()
    print('Negative ΔFOS = the wider variant finds a less favorable surface (expected).')
    print(f'Full output: {OUTFILE}')


if __name__ == '__main__':
    main()