"""Cached native-reference YUV -> standard H.264 baseline; no latent encoding claim."""
import subprocess,numpy as np

def encode_decode(raw,path,qp):
    assert len(raw)==17*480*832*3//2
    subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i','pipe:0','-c:v','libx264','-preset','medium','-qp',str(qp),'-g','17','-bf','2','-refs','3','-threads','4','-color_primaries','bt709','-colorspace','bt709','-color_trc','bt709','-color_range','tv','-f','h264',str(path)],input=raw,check=True)
    decoded=subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(path),'-f','rawvideo','-pix_fmt','yuv420p','pipe:1'],capture_output=True,check=True).stdout
    rgb=subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','yuv420p','-s','832x480','-r','16','-i','pipe:0','-vf','scale=in_range=limited:out_range=full:in_color_matrix=bt709','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],input=decoded,capture_output=True,check=True).stdout
    assert len(decoded)==len(raw) and len(rgb)==17*480*832*3
    return decoded,np.frombuffer(rgb,dtype=np.uint8).reshape(17,480,832,3).copy()
