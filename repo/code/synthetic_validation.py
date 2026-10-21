import sys,os,numpy as np,cv2,warnings;warnings.filterwarnings('ignore')
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE);sys.argv=['x'];import knitqr_analyze as kq
P,B=kq.P,kq.B
T=kq.truth_from_chart(os.path.join(ROOT,'design','QR_knit_chart_74x111_as_seen.png'),2,3,4)
N=T.shape[0]; M=N+2*B
rng=np.random.default_rng(7)
S=20  # px per module in render
des=np.kron(1-np.pad(T,B),np.ones((S,S))).astype(np.float32)  # 1=light
def make_u(amp):
    # smooth field in module units on canvas [0,M]
    comps=[]
    for _ in range(4):
        kx,ky=rng.uniform(0.5,2.0,2)*2*np.pi/M; ph=rng.uniform(0,2*np.pi,2); a=rng.normal(0,1,2)
        comps.append((kx,ky,ph,a))
    bend=rng.normal(0,1,2)
    def u(x,y):
        ux=np.zeros_like(x,dtype=float); uy=np.zeros_like(x,dtype=float)
        for kx,ky,ph,a in comps:
            s=np.sin(kx*x+ph[0])*np.cos(ky*y+ph[1]); ux+=a[0]*s; uy+=a[1]*np.cos(kx*x+ph[1])*np.sin(ky*y+ph[0])
        c=(x-M/2)/M; d=(y-M/2)/M
        ux+=bend[0]*d*d*4; uy+=bend[1]*c*c*4
        return ux,uy
    # normalise later by scale
    return u
rows=[]
lat=kq.lattice(N)
for i,amp in enumerate(np.repeat([0.0,0.1,0.2,0.3,0.4,0.5,0.6],5)):
    u0=make_u(1)
    # scale so that non-projective rms at lattice ~ amp*something; use raw scale then measure truth
    sc=amp
    u=lambda x,y,u0=u0,sc=sc:tuple(sc*v for v in u0(x,y))
    # render deformed canvas: pixel q (module coords) shows design at p=q-u(q)
    H_,W_=M*S,M*S
    yy,xx=np.mgrid[0:H_,0:W_].astype(np.float32)
    qx,qy=(xx+0.5)/S,(yy+0.5)/S
    ux,uy=u(qx,qy)
    mapx=((qx-ux)*S-0.5).astype(np.float32); mapy=((qy-uy)*S-0.5).astype(np.float32)
    img=cv2.remap(des,mapx,mapy,cv2.INTER_LINEAR,borderValue=1.0)
    img=cv2.GaussianBlur(img,(0,0),1.2)
    # colour: green/ivory + noise
    g=(img*200+40+rng.normal(0,8,img.shape)).clip(0,255).astype(np.uint8)
    pad=200; g=cv2.copyMakeBorder(g,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=235)
    # camera homography
    h,w=g.shape; src=np.float32([[0,0],[w,0],[w,h],[0,h]]); jit=rng.uniform(-0.06,0.06,(4,2))*w
    Hc=cv2.getPerspectiveTransform(src,(src+jit).astype(np.float32)); out=cv2.warpPerspective(g,Hc,(w,h),borderValue=235)
    p=os.path.join(ROOT,'out','synthetic',f'syn_{i:02d}.jpg')
    os.makedirs(os.path.dirname(p),exist_ok=True); cv2.imwrite(p,cv2.cvtColor(out,cv2.COLOR_GRAY2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
    # truth at lattice: design point p (module coords of canvas centre of lattice cell): kq uses pixel mx*P in canvas -> module coord mx
    pts=np.array([(mx,my) for my in lat for mx in lat],float)
    q=pts.copy()
    for _ in range(30):
        ux,uy=u(q[:,0],q[:,1]); q=np.stack([pts[:,0]+ux,pts[:,1]+uy],1)
    Ht,_=cv2.findHomography(q.astype(np.float32),pts.astype(np.float32),0)
    pr=cv2.perspectiveTransform(q[None].astype(np.float32),Ht)[0]
    res=np.linalg.norm(pr-pts,axis=1); d=pts-B-0.5; x,y=d[:,0],d[:,1]
    fin=((x<=8)&(y<=8))|((x>=20)&(y<=8))|((x<=8)&(y>=20))
    t_rms=float(np.sqrt((res**2).mean())); t_fin=float(res[fin].max())
    r=kq.analyze(p,T,os.path.join(ROOT,'out','synthetic_analysis'))
    if '_lat' in r:
        dst,prr,_=r['_lat']; dd=dst/P-B-0.5; rr=np.linalg.norm(prr/P-B-0.5-dd,axis=1); xx_,yy_=dd[:,0],dd[:,1]
        f2=((xx_<=8)&(yy_<=8))|((xx_>=20)&(yy_<=8))|((xx_<=8)&(yy_>=20))
        e_rms=r['resid_rms_mod']; e_fin=float(rr[f2].max()); me=r['module_err_pct']
    else: e_rms=e_fin=me=np.nan
    rows.append((i,amp,t_rms,e_rms,t_fin,e_fin,me)); print(rows[-1],flush=True)
import pandas as pd
pd.DataFrame(rows,columns=['i','amp','true_rms','est_rms','true_fin','est_fin','moderr']).to_csv(os.path.join(ROOT,'data','synthetic_validation.csv'),index=False)
print('DONE')
