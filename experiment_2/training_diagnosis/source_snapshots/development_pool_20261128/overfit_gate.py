from common import *
import numpy as np
rows=[];cal=[]
for factor in (.1,.3,1,3,10):
    name=f'O2_R3_identity_f{factor:g}';x=json.loads((ROOT/'results/E2_1c'/name/'evaluation_step1000.json').read_text());rows.append(x);c=json.loads((ROOT/'results/E2_1c'/name/'calibration.json').read_text());cal.append(c)
corrd=float(np.corrcoef(np.log([x['lambda_z'] for x in rows]),[x['latent_mse'] for x in rows])[0,1]);corrr=float(np.corrcoef(np.log([x['lambda_z'] for x in rows]),[x['actual_bytes'] for x in rows])[0,1]);passed=corrd<-.5 and corrr>.5 and all(x['independent_process_decode'] and x['psnr_y']>30 for x in rows)
save(ROOT/'results/E2_1c/gate.json',dict(pass_=bool(passed),**{'pass':bool(passed)},distortion_loglambda_correlation=corrd,bytes_loglambda_correlation=corrr,scope='one training video only; latent MSE and PSNR evaluated separately',variant='R3_identity',quant_step=.1,symbol_limit=255))
csvsave(ROOT/'results/E2_1c/lambda_calibration.csv',cal);csvsave(ROOT/'results/E2_1c/single_video_overfit.csv',[{k:x[k] for k in ['name','step','lambda_z','latent_mse','actual_bytes','main_bytes','hyper_bytes','metadata_bytes','psnr_y','psnr_rgb','last4_psnr_y','independent_process_decode']} for x in rows]);print('E2.1c gate',passed,corrd,corrr,flush=True)
assert passed,'Do not launch scale training until single-video learning works'
