import os,glob,cv2,numpy as np,pandas as pd,zxingcpp
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
# usage: python decoder_reread.py  (photos in ../photos/, WeChat models in ./wechat_models/)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
W=os.path.join(HERE,'wechat_models')+'/'   # detect.prototxt/.caffemodel, sr.prototxt/.caffemodel from opencv_3rdparty
wc=cv2.wechat_qrcode_WeChatQRCode(W+'detect.prototxt',W+'detect.caffemodel',W+'sr.prototxt',W+'sr.caffemodel')
URL='zrr.kr/ZFyyPM'
man=pd.read_csv(os.path.join(ROOT,'data','photo_manifest.csv'))
L=[(r.ds,r.fab,r.st,os.path.join(ROOT,'photos',r.file),int(r.phone)) for r in man.itertuples()]
def variants(img):
    h,w=img.shape[:2]; out=[('orig',img)]
    for m in (2000,1400,1000,700):
        if max(h,w)>m:
            f=m/max(h,w); out.append((f's{m}',cv2.resize(img,None,fx=f,fy=f,interpolation=cv2.INTER_AREA)))
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out.append(('clahe',cv2.createCLAHE(3,(8,8)).apply(g)))
    out.append(('blur',cv2.GaussianBlur(g,(0,0),2)))
    return out
def ok(t): return t is not None and URL in t
R=[]
for ds,fab,s,p,rec in L:
    img=cv2.imread(p); res={}
    V=variants(img); V+= [('flip_'+n,cv2.flip(v,1)) for n,v in V]
    def z(v):
        for b in (zxingcpp.Binarizer.LocalAverage,zxingcpp.Binarizer.GlobalHistogram):
            if any(ok(x.text) for x in zxingcpp.read_barcodes(v,formats=zxingcpp.BarcodeFormat.QRCode,try_rotate=True,try_downscale=True,binarizer=b)): return 1
        return 0
    res['zxing']=max(z(v) for _,v in V)
    res['zbar']=max(int(any(ok(d.data.decode(errors='ignore')) for d in pyzbar.decode(v,symbols=[ZBarSymbol.QRCODE]))) for _,v in V)
    def cvd(v):
        try: return int(ok(cv2.QRCodeDetector().detectAndDecode(v)[0]))
        except: return 0
    res['opencv']=max(cvd(v) for _,v in V)
    def wcd(v):
        try: return int(any(ok(t) for t in wc.detectAndDecode(v)[0]))
        except: return 0
    res['wechat']=max(wcd(v) for _,v in V[:len(V)//2]+[V[len(V)//2]])
    R.append(dict(ds=ds,fab=fab,st=s,file=os.path.basename(p),phone=rec,**res)); print(R[-1],flush=True)
pd.DataFrame(R).to_csv(os.path.join(ROOT,'data','decoder_reread.csv'),index=False)
