"""Display-only exports. Formal quality always precedes MP4 encoding."""
import json
import subprocess
import numpy as np
import runtime as r
from run import ROOT,E0,MODEL,save
from codec.lvc import decode

def main():
    assert (ROOT/'results/ALL_MAIN_MEASUREMENTS_COMPLETE').exists()
    uniform=json.loads((ROOT/'results/E1_0/rows.json').read_text())
    mixed=json.loads((ROOT/'results/E1_2/mixed_precision_results.json').read_text())
    reuse=json.loads((ROOT/'results/E1_3/skip_probe_results.json').read_text())
    baseline=json.loads((E0/'results/rows.json').read_text())+json.loads((E0/'results/validation_rows.json').read_text())
    manifest=json.loads((E0/'results/validation_manifest.json').read_text())
    sources={s['sample']:s['latent'] for s in r.samples};sources.update({s['sample_id']:s['final_latent'] for s in manifest})
    r.vae=r.WanVAE(vae_pth=str(MODEL),device='cuda:0');exports=[]
    comparisons=list(__import__('csv').DictReader((ROOT/'results/E1_2/actual_rate_matched_comparison.csv').open()))
    policies=json.loads((ROOT/'configs/frozen_mixed_policies.json').read_text())['policies']
    deployable=[p for p in policies if p['method']!='M1']
    def development_score(policy):
        vals=[float(v['gain_db']) for v in comparisons if v['split']=='development' and v['gain_db'] and json.loads(v['policy'])==policy]
        return float(np.mean(vals)) if vals else -1e30
    chosen_policy=max(deployable,key=development_score)
    for name in ('sample_0000','sample_1004','sample_1032'):
        folder=ROOT/'videos'/name;folder.mkdir(parents=True,exist_ok=True);panels=[]
        z=r.torch.load(sources[name],weights_only=True,map_location='cpu').numpy()[None]
        reference,_,_,_=r.pixels(z)
        def panel(method,video,row=None,note=''):
            path=folder/(method+'.mp4');label=folder/(method+'.txt')
            if row:
                labeltext=f"{method} {note}\nPSNR-Y {row['psnr_y']:.2f} dB\n"
                if isinstance(row.get('total_bytes'),int):
                    enc=row.get('encode_s',{}).get('mean',row.get('encode_s_single',0))*1000
                    labeltext+=f"{row['total_bytes']} Bytes {row['bitrate_kbps']:.1f} kbps enc {enc:.2f} ms"
                else:labeltext+='Actual skip bytes / bitrate / encoding time NOT RUN'
            else:labeltext=f'{method} {note}\nReference; PSNR infinite\nCodec bytes and encoding time N/A'
            label.write_text(labeltext)
            r.cmd(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-r','16','-i','pipe:0','-vf',f'drawtext=textfile={label}:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7:x=8:y=8','-pix_fmt','yuv420p','-c:v','libx264','-crf','18',str(path)],video.tobytes())
            panels.append(path)
        panel('reference',reference)
        old=next(v for v in baseline if v['sample']==name and v['method']=='x264_medium_28')
        rgb=r.cmd(['ffmpeg','-v','error','-i',str(E0/'results'/name/'x264_medium_28.264'),'-vf','scale=in_color_matrix=bt709:in_range=limited:out_range=full','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        panel('h264_QP28',np.frombuffer(rgb,dtype='uint8').reshape(17,480,832,3),old,'E0 medium')
        for bits in (4,5,6,8):
            row=next(v for v in uniform if v['sample']==name and v['method']==f'Q{bits}')
            rec,_,_,_=r.pixels(decode(ROOT/'results/E1_0'/name/f'Q{bits}.lvc'));panel(f'uniform_INT{bits}',rec,row)
        candidates=[v for v in mixed if v['sample']==name and v['policy']['method'] in ('M2','M2_three','M3')]
        best=next(v for v in candidates if v['policy']==chosen_policy)
        stream=ROOT/'results/E1_2'/name/('bits_'+''.join(map(str,best['channel_bits']))+'.lvc')
        rec,_,_,_=r.pixels(decode(stream));panel('mixed_deployable',rec,best,best['policy']['method']+' development-selected policy')
        oracle=[v for v in mixed if v['sample']==name and v['policy']['method']=='M4_approximate_oracle']
        if oracle:
            row=max(oracle,key=lambda v:v['psnr_y']);stream=ROOT/'results/E1_2'/name/('bits_'+''.join(map(str,row['channel_bits']))+'.lvc')
            rec,_,_,_=r.pixels(decode(stream));panel('approx_oracle',rec,row,'OFFLINE search excluded')
        cfg=dict(operation='reuse',ratio=.1,strategy='low',mode='closed_loop')
        row=next(v for v in reuse if v['sample']==name and v['block_size']==4 and v['configuration']==cfg)
        rec=decode(ROOT/'results/E1_0'/name/'Q8.lvc');coords=[(y,x) for y in range(0,60,4) for x in range(0,104,4)]
        for t in range(1,5):
            energies=[np.mean((z[0,:,t,y:y+4,x:x+4]-z[0,:,t-1,y:y+4,x:x+4])**2) for y,x in coords]
            for i in np.argsort(energies,kind='stable')[:round(.1*len(coords))]:
                y,x=coords[i];rec[0,:,t,y:y+4,x:x+4]=rec[0,:,t-1,y:y+4,x:x+4]
        video,_,_,_=r.pixels(rec)
        q=r.quality(reference,video,r.yuv(reference),r.yuv(video));assert abs(q['psnr_y']-row['psnr_y'])<1e-6
        panel('reuse_low10_closed',video,row,'THEORETICAL RATE ONLY')
        args=['ffmpeg','-y','-v','error']
        for path in panels:args+=['-i',str(path)]
        filters=[f'[{i}:v]scale=416:240[v{i}]' for i in range(len(panels))]
        layout='|'.join(f'{(i%3)*416}_{(i//3)*240}' for i in range(len(panels)))
        filters.append(''.join(f'[v{i}]' for i in range(len(panels)))+f'xstack=inputs={len(panels)}:layout={layout}:fill=black[out]')
        comparison=folder/'comparison.mp4';r.cmd(args+['-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','18',str(comparison)])
        for path in panels+[comparison]:
            result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
            assert int(result['nb_read_frames'])==17 and result['r_frame_rate']=='16/1'
            if path!=comparison:assert (result['width'],result['height'])==(832,480)
            exports.append(dict(path=str(path.relative_to(ROOT)),**result,display_only=True))
        print(name,'DONE',flush=True)
    save(ROOT/'results/video_manifest.json',exports)

if __name__=='__main__':main()
