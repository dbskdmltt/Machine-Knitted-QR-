# Finder-region displacement: max vs RMS vs 90th percentile (photos in ../photos/)
import sys,os,csv,numpy as np,warnings;warnings.filterwarnings('ignore')
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.argv=['x']; import knitqr_analyze as kq
P,B=kq.P,kq.B; T=kq.truth_from_chart(os.path.join(ROOT,'design','QR_knit_chart_74x111_as_seen.png'),2,3,4)
w=csv.writer(open(os.path.join(ROOT,'data','finder_sensitivity_raw.csv'),'w'))
w.writerow(['ds','fab','st','file','ncc','rms','finmax','finp90','finrms','gridp90'])
for r in csv.DictReader(open(os.path.join(ROOT,'data','photo_manifest.csv'))):
    p=os.path.join(ROOT,'photos',r['file'])
    rec=kq.analyze(p,T,os.path.join(ROOT,'out','finder'))
    if '_lat' not in rec: w.writerow([r['ds'],r['fab'],r['st'],r['file'],'ERR']); continue
    dst,pr,_=rec['_lat']; d=dst/P-B-0.5; o=pr/P-B-0.5; res=np.linalg.norm(o-d,axis=1); x,y=d[:,0],d[:,1]
    fin=((x<=8)&(y<=8))|((x>=20)&(y<=8))|((x<=8)&(y>=20))   # three 9x9-module finder regions
    w.writerow([r['ds'],r['fab'],r['st'],r['file'],rec.get('match_ncc_mean'),rec['resid_rms_mod'],
                round(float(res[fin].max()),3),round(float(np.percentile(res[fin],90)),3),
                round(float(np.sqrt((res[fin]**2).mean())),3),round(float(np.percentile(res,90)),3)])
