"""Write four PyMOL batch scripts using aligned full-atom case exports."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from common import OUT

HEADER = '''python
import csv
import math
from pathlib import Path
from pymol import cmd
from pymol.cgo import CYLINDER, CONE

cmd.bg_color('white')
cmd.set('ray_opaque_background', 0)
cmd.set('antialias', 2)
cmd.set('orthoscopic', 1)
cmd.set('depth_cue', 0)
cmd.set('specular', 0.2)
cmd.set('cartoon_fancy_helices', 1)
cmd.set('cartoon_side_chain_helper', 1)
cmd.set('ray_trace_mode', 1)

BLUE = (0.58, 0.64, 0.72)
GRAY = (0.80, 0.84, 0.88)
ROSE = (0.882, 0.114, 0.282)
TEAL = (0.008, 0.518, 0.545)
AMBER = (0.847, 0.467, 0.024)

def rgb(hexcode):
    h=hexcode.lstrip('#')
    return tuple(int(h[i:i+2],16)/255.0 for i in (0,2,4))

def blend(a,b,t):
    return tuple(a[i]*(1-t)+b[i]*t for i in range(3))

def scale(v, anchors):
    if v<=anchors[0][0]: return rgb(anchors[0][1])
    if v>=anchors[-1][0]: return rgb(anchors[-1][1])
    for (lo,c1),(hi,c2) in zip(anchors, anchors[1:]):
        if v<=hi: return blend(rgb(c1),rgb(c2),(v-lo)/(hi-lo))

DISP = [(0,'#F8FAFC'),(.5,'#FEF08A'),(1,'#F97316'),(2.5,'#BE123C')]
BG = [(0,'#E0F2FE'),(.5,'#0284C7')]
EXCESS = [(0,'#FFFFFF'),(2.5,'#E11D48')]
ERROR = [(0,'#F8FAFC'),(.5,'#FEF08A'),(1,'#F97316'),(2.5,'#BE123C')]

def read_rows(path):
    with open(path) as f: return list(csv.DictReader(f))

def paint(obj, rows, field, palette=DISP):
    for i, row in enumerate(rows,1):
        name='v'+str(i)
        cmd.set_color(name,scale(float(row[field]),palette))
        cmd.color(name, obj+' and resi '+str(i))

def save(path):
    cmd.ray(2400,1800)
    cmd.png(path, width=2400, height=1800, dpi=300, ray=0)

def arrow(start,end,name,color=AMBER,radius=.14):
    vec=[end[j]-start[j] for j in range(3)]
    length=math.sqrt(sum(v*v for v in vec))
    if length < .1: return
    shoulder=[start[j]+.78*vec[j] for j in range(3)]
    obj=[CYLINDER,*start,*shoulder,radius,*color,*color,
         CONE,*shoulder,*end,radius*2.0,0.,*color,*color,1.,0.]
    cmd.load_cgo(obj,name)

def rings(center, radii):
    for radius in radii:
        cgo=[]
        for n in range(0,60,3):
            a=2*math.pi*n/60
            b=2*math.pi*(n+2)/60
            start=(center[0]+radius*math.cos(a),center[1]+radius*math.sin(a),center[2])
            end=(center[0]+radius*math.cos(b),center[1]+radius*math.sin(b),center[2])
            cgo.extend([CYLINDER,*start,*end,.055,.6,.63,.7,.6,.63,.7])
        cmd.load_cgo(cgo,'ring'+str(radius).replace('.','_'))

'''


def case1_code(path: Path) -> str:
    base = str(path.resolve())
    return f'''root=Path({base!r})
rows=read_rows(root/'residues.csv')
mut_pos={int(pd.read_csv(path/'residues.csv').query('radii==0').position.iloc[0])}
cmd.load(str(root/'wt.pdb'),'wt')
cmd.load(str(root/'mut.pdb'),'mut')
cmd.show('cartoon','wt or mut')
cmd.orient('wt or mut')
view=cmd.get_view()
cmd.hide('everything','all')
cmd.show('cartoon','wt or mut')
cmd.color('gray70','wt')
cmd.set_color('wtblue',BLUE)
cmd.set_color('mutgray',GRAY)
cmd.set_color('perturbed',ROSE)
cmd.color('wtblue','wt')
cmd.color('mutgray','mut')
cmd.set('cartoon_transparency',.45,'wt')
for i,row in enumerate(rows,1):
    if float(row['true'])>1.0: cmd.color('perturbed','mut and resi '+str(i))
cmd.show('sticks',f'wt and resi {{mut_pos}}')
cmd.show('sticks',f'mut and resi {{mut_pos}}')
cmd.color('marine',f'wt and resi {{mut_pos}}')
cmd.color('firebrick',f'mut and resi {{mut_pos}}')
center=cmd.get_atom_coords(f'wt and resi {{mut_pos}} and name CA')
rings(center,(4.,8.,12.))
strong=sorted(range(len(rows)), key=lambda j:float(rows[j]['true']), reverse=True)[:8]
for j in strong:
    x=cmd.get_atom_coords(f'wt and resi {{j+1}} and name CA')
    y=cmd.get_atom_coords(f'mut and resi {{j+1}} and name CA')
    arrow(x,y,'true_arrow_'+str(j))
cmd.set_view(view)
save(str(root/'overlay.png'))
cmd.zoom('mut and resi '+'+'.join(str(j+1) for j in strong), buffer=4.0)
save(str(root/'zoom.png'))
cmd.delete('all')
for field in ('true','musrnet','geometry','coordinate','af2'):
    cmd.load(str(root/'wt_'+field+'.pdb'),'model')
    cmd.show('cartoon','model')
    paint('model',rows,field)
    cmd.show('spheres',f'model and resi {{mut_pos}} and name CA')
    cmd.set('sphere_scale',.35,'model')
    if field=='coordinate':
        for j in strong:
            origin=cmd.get_atom_coords(f'model and resi {{j+1}} and name CA')
            delta=[float(rows[j][k]) for k in ('vector_x','vector_y','vector_z')]
            arrow(origin,[origin[k]+delta[k] for k in range(3)],'pred_arrow_'+str(j),TEAL,.08)
    cmd.set_view(view)
    save(str(root/(field+'.png')))
    cmd.delete('all')
'''


def radial_code(path: Path) -> str:
    root = str(path.resolve())
    return f'''root=Path({root!r})
rows=read_rows(root/'residues.csv')
mut_pos=next(i for i,row in enumerate(rows,1) if float(row['radii']) < 1.e-4)
cmd.load(str(root/'wt.pdb'),'wt')
cmd.color('gray75','wt')
cmd.show('cartoon','wt')
cmd.set('cartoon_transparency',.18,'wt')
cmd.create('truth','wt')
cmd.create('pred','wt')
cmd.hide('everything','truth or pred')
truth=[i for i,row in enumerate(rows,1) if float(row['true'])>1.0]
pred=[i for i,row in enumerate(rows,1) if float(row['score'])>0.1]
cmd.set_color('truthgold',(0.98,0.70,0.11))
cmd.set_color('predred',ROSE)
for i in truth:
    cmd.show('spheres',f'truth and resi {{i}} and name CA')
for i in pred:
    cmd.show('spheres',f'pred and resi {{i}} and name CA')
cmd.color('truthgold','truth')
cmd.color('predred','pred')
cmd.set('sphere_scale',.7,'truth')
cmd.set('sphere_scale',.38,'pred')
cmd.orient('wt')
view=cmd.get_view()
center=cmd.get_atom_coords(f'wt and resi {{mut_pos}} and name CA')
rho=max((float(rows[i-1]['radii']) for i in truth),default=0.)
rho_hat=max((float(rows[i-1]['radii']) for i in pred),default=0.)
rings(center,sorted(set((8.,round(rho,2),round(rho_hat,2)))))
cmd.set_view(view)
save(str(root/'radial.png'))
'''


def mechanism_code(path: Path) -> str:
    base = str(path.resolve())
    return f'''root=Path({base!r})
rows=read_rows(root/'residues.csv')
cmd.load(str(root/'wt.pdb'),'ref')
cmd.orient('ref')
view=cmd.get_view()
cmd.delete('ref')
for field,palette in (('background',BG),('gated_excess',EXCESS),
                      ('musrnet',DISP),('true',DISP),
                      ('abs_error_full',ERROR),('abs_error_global',ERROR)):
    cmd.load(str(root/'wt_'+field+'.pdb'),'model')
    cmd.show('cartoon','model')
    paint('model',rows,field,palette)
    cmd.set_view(view)
    save(str(root/(field+'.png')))
    if field.startswith('abs_error'):
        mut_pos=next(i for i,row in enumerate(rows,1) if float(row['radii'])<1.e-4)
        cmd.select('near', f'byres (model within 8 of (model and resi {{mut_pos}} and name CA))')
        cmd.zoom('near', buffer=4.0)
        save(str(root/(field+'_zoom.png')))
    cmd.delete('all')
'''


def main():
    folder = OUT / 'pymol'
    folder.mkdir(parents=True, exist_ok=True)
    scripts = {
        'case1.pml': case1_code(OUT / 'cases/case1_comparison'),
        'case2_local.pml': radial_code(OUT / 'cases/case2_local'),
        'case3_nonlocal.pml': radial_code(OUT / 'cases/case3_nonlocal'),
        'case4_mechanism.pml': mechanism_code(OUT / 'cases/case4_mechanism'),
    }
    for filename, body in scripts.items():
        (folder / filename).write_text(HEADER + body + '\npython end\n')
        print('Wrote', folder / filename)


if __name__ == '__main__':
    main()
