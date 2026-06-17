import hyrcan as hy
import math, csv, os, shutil

INFILE = 'pilot_50.csv'
METHOD = 'Spencer'

# Variante incercate in ordine pana la rezolvare
REPAIR_VARIANTS = [
    {'ext_factor': None,  'depth_factor': None},
    {'ext_factor': 2.0,   'depth_factor': None},
    {'ext_factor': None,  'depth_factor': 2.0},
    {'ext_factor': 1.5,  'depth_factor': 1.5},   # both_1_5x
    {'ext_factor': 2.0,  'depth_factor': 2.0},   # both_2x
    {'ext_factor': 3.0,  'depth_factor': 3.0},   # both_3x — last resort
]


# ── Geometrie (identica cu scriptul de generare) ──────────────────────────────
def model_dims(H, angle, ext_factor=None, depth_factor=None):
    slope        = math.tan(angle * math.pi / 180)
    slope_length = H / slope
    extension    = max(1.5 * H, slope_length * 0.5)
    depth        = max(H, 0.5 * slope_length)
    if ext_factor:
        extension *= ext_factor
    if depth_factor:
        depth *= depth_factor
    return slope, slope_length, extension, depth


def initialize_model(H, angle, phi, c, gamma, ext_factor=None, depth_factor=None):
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
    return cmd


# ── QC suprafata critica ──────────────────────────────────────────────────────
def find_intersection(cx, cy, r, y):
    if abs(cy - y) > r:
        return None
    dx = math.sqrt(max(r**2 - (cy - y)**2, 0))
    return (cx - dx, cx + dx)


def check_boundary(cx, cy, r, H, angle, ext_factor=None, depth_factor=None):
    _, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    tol = 0.01 * H
    flags = []

    gi = find_intersection(cx, cy, r, 0)
    if gi and abs(gi[0] - (-depth)) < tol:
        flags.append('stanga_limita_inf')

    ti = find_intersection(cx, cy, r, H)
    if ti and abs(ti[1] - (slope_length + extension)) < tol:
        flags.append('dreapta_limita_sup')

    if cy - r < -depth:
        flags.append('sub_domeniu')

    return bool(flags), flags


# ── Calcul ────────────────────────────────────────────────────────────────────
def run_hyrcan(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None):
    """Returneaza (fos, cx, cy, r, slope_length, extension, depth) sau None."""
    try:
        _, slope_length, extension, depth = model_dims(
            H, angle, ext_factor, depth_factor)
        hy.command(initialize_model(H, angle, phi, c, gamma, ext_factor, depth_factor))
        hy.command("compute('silence')")
        fos    = hy.min_fos(METHOD)
        center = hy.surf_center_min_fos(METHOD)
        cx, cy = center[0], center[1]
        r      = hy.surf_radius_min_fos(METHOD)
        return fos, cx, cy, r, slope_length, extension, depth
    except Exception as e:
        print(f'    EROARE Hyrcan: {e}')
        import traceback; traceback.print_exc()
        return None


# ── CSV helpers ───────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Reparare ──────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INFILE):
        print(f'Fisier negasit: {INFILE}')
        return

    backup = INFILE.replace('.csv', '_backup.csv')
    shutil.copy(INFILE, backup)
    print(f'Backup creat: {backup}\n')

    rows, fieldnames = read_csv(INFILE)

    # adauga coloanele de metadate daca lipsesc
    for col in ['repair_variant', 'excluded']:
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        row.setdefault('repair_variant', '')
        row.setdefault('excluded', '')

    # identifica problemele
    problems = [(i, r) for i, r in enumerate(rows)
                if str(r.get('At_Boundary', '')).strip().lower() == 'true'
                and to_float(r.get('FOS')) is not None]

    # raporteaza si erorile solver (FOS gol) — nu le repara, doar le marcheaza
    solver_errors = [(i, r) for i, r in enumerate(rows)
                     if to_float(r.get('FOS')) is None]
    for i, r in solver_errors:
        rows[i]['excluded'] = 'eroare_solver'

    if not problems:
        print('Niciun caz At_Boundary=True de reparat.')
        if solver_errors:
            print(f'{len(solver_errors)} cazuri cu eroare solver — marcate excluded.')
            write_csv(INFILE, rows, fieldnames)
        return

    print(f'{len(problems)} cazuri At_Boundary=True de reparat.')
    if solver_errors:
        print(f'{len(solver_errors)} cazuri cu eroare solver — marcate excluded, nereparate.')
    print()

    n_repaired = 0
    n_unresolved = 0

    for i, row in problems:
        angle = to_float(row['angle']); phi   = to_float(row['phi'])
        c     = to_float(row['c']);     gamma = to_float(row['gamma'])
        H     = to_float(row['H']);     fos_orig = to_float(row['FOS'])

        print(f'─── Rand {i+1}: angle={angle:.1f}° φ={phi:.1f}° '
              f'c={c:.1f} γ={gamma:.1f} H={H:.1f}m  FOS_orig={fos_orig:.4f}')

        resolved = False
        for vp in REPAIR_VARIANTS:
            ef = vp['ext_factor']
            df = vp['depth_factor']
            label = f"ext{ef or 1}x_depth{df or 1}x"
            print(f'  Incerc {label} ...', end=' ')

            result = run_hyrcan(angle, phi, c, gamma, H, ef, df)
            if result is None:
                print('eroare solver.')
                continue

            fos, cx, cy, r, sl, ext, depth = result
            still_boundary, flags = check_boundary(cx, cy, r, H, angle, ef, df)
            delta = fos - fos_orig

            if not still_boundary:
                print(f'REZOLVAT. FOS={fos:.4f} (ΔFOS={delta:+.4f})')
                rows[i].update({
                    'FOS':            round(fos, 5),
                    'Center_X':       round(cx, 4),
                    'Center_Y':       round(cy, 4),
                    'Radius':         round(r, 4),
                    'At_Boundary':    False,
                    'slope_length':   round(sl, 3),
                    'extension':      round(ext, 3),
                    'depth':          round(depth, 3),
                    'repair_variant': label,
                    'excluded':       '',
                })
                resolved = True
                n_repaired += 1
                break
            else:
                print(f'still_boundary ({", ".join(flags)}). '
                      f'FOS={fos:.4f} (ΔFOS={delta:+.4f})')

        if not resolved:
            print('  Nicio varianta nu a rezolvat — marcat excluded.')
            rows[i]['excluded'] = 'nerezolvat'
            n_unresolved += 1

        print()

    write_csv(INFILE, rows, fieldnames)

    # ── Sumar final ───────────────────────────────────────────────────────────
    total    = len(rows)
    valid    = sum(1 for r in rows
                   if str(r.get('At_Boundary', '')).strip().lower() == 'false'
                   and not r.get('excluded'))
    excluded = sum(1 for r in rows if r.get('excluded'))

    print('=' * 55)
    print('SUMAR REPARARE')
    print('=' * 55)
    print(f'Total randuri:               {total}')
    print(f'Valide pentru antrenament:   {valid}')
    print(f'Reparate cu succes:          {n_repaired}')
    print(f'Nerezolvate (excluded):      {n_unresolved}')
    print(f'Erori solver (excluded):     {len(solver_errors)}')
    print(f'\nCSV actualizat: {INFILE}')
    print(f'Original intact: {backup}')
    print('\nFiltrare la antrenament: pastreaza doar randurile cu excluded gol.')


if __name__ == '__main__':
    main()