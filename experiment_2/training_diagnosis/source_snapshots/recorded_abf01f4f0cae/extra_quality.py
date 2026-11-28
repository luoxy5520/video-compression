"""Full-sequence SSIM-RGB and labeled qualitative video; no perceptual weights downloaded."""
from pixels import *
from diagnostic_codec import load
from temporal_entropy_model import load_temporal
from pytorch_msssim import ssim
from PIL import Image,ImageDraw
out=ROOT/'results/E2_1e';v=vae();base_records=json.loads((ROOT/'results/E2_1d/D16_f1_seed20261127/development_step10000.json').read_text());allrecords=json.loads((E2/'dataset_manifest.json').read_text())['records'];rows=[]
for idx,r in enumerate(base_records):
    record=next(a for a in allrecords if a['path']==r['video']);x=tensor(record)
    with torch.no_grad():reference=rgb(decode(v,x))
    images={'Original':reference}
    configs=[('V0',ROOT/'results/E2_1d/D16_f1_seed20261127','10000'),('V1',out/'V1_D16','1000'),('V2',out/'V2_D16','1000')]
    for label,folder,step in configs:
        modelpath=folder/f'deploy_step{step}.pt';m,c=load(modelpath);recs=[]
        for t in range(5):recs.append(read_stream(folder/f'step{step}_dev{idx}_t{t}.lvc2',m,sha(modelpath)))
        with torch.no_grad():image=rgb(decode(v,torch.cat(recs)))
        images[label]=image;a=torch.from_numpy(reference.copy()).permute(0,3,1,2).float();b=torch.from_numpy(image.copy()).permute(0,3,1,2).float();score=ssim(a,b,data_range=255,size_average=True).item();rows.append(dict(video=r['video'],method=label,SSIM_RGB=score,scope='pytorch_msssim1.0.0 RGB8bit,data_range255,mean over17 full-sequence frames',LPIPS='NOT RUN: no verified perceptual weights',DISTS='NOT RUN: no verified perceptual weights'))
    # Display encoding only, never used as quantitative reference.
    panels=[]
    for label,image in images.items():
        frames=[]
        for frame in image:
            canvas=Image.fromarray(frame);d=ImageDraw.Draw(canvas);d.rectangle((0,0,260,34),fill='black');d.text((8,8),label+' (display only)',fill='white');frames.append(np.asarray(canvas))
        panels.append(np.stack(frames))
    movie=np.concatenate(panels,axis=2)
    subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','3328x480','-r','16','-i','pipe:0','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p',str(out/f'qualitative_video_comparison_dev{idx}.mp4')],input=movie.tobytes(),check=True)
    save(out/'extra_quality.json',rows)
csvsave(out/'extra_quality.csv',rows)
