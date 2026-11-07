"""Read only the Oracle I payload, then make neutral fixed-config P placeholders."""
import argparse
import struct
from pathlib import Path

HEADER=struct.Struct("<8s14I")
MB=struct.Struct("<IIhhhhh4b32h16b")


def read_exact(f,n):
    b=f.read(n)
    if len(b)!=n:
        raise ValueError("Truncated I payload")
    return b


def isolate(source,output,frames):
    if frames<2:
        raise ValueError("At least one P frame required")
    with Path(source).open("rb") as f:
        header=list(HEADER.unpack(read_exact(f,HEADER.size)))
        if tuple(header[:12])!=(b"JMZV1\0\0\0",7,832,480,1560,6,1,0,66,40,1,3):
            raise ValueError("Expected restricted B1 Baseline/CAVLC layout")
        if header[13]!=MB.size or header[14]!=0:
            raise ValueError("B1 record layout mismatch")
        header[12]=frames*1560
        i_records=[]
        for addr in range(1560):
            raw=read_exact(f,MB.size)
            mb=MB.unpack(raw)
            if mb[:5]!=(0,addr,2,10,28):
                raise ValueError("Expected raster-ordered QP28 I16 records")
            i_records.append(raw)
            for _ in range(27):
                n=read_exact(f,1)
                i_records.extend((n,read_exact(f,5*n[0])))
            i_records.append(read_exact(f,384))
        # No P records are read, even if the source contains them.
        source_bytes_read=f.tell()
    with Path(output).open("xb") as f:
        f.write(HEADER.pack(*header))
        f.writelines(i_records)
        for frame in range(1,frames):
            for addr in range(1560):
                f.write(MB.pack(frame,addr,0,0,28,0,0,*([0]*52)))
    return source_bytes_read


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--oracle-i",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=17)
    a=p.parse_args()
    print(isolate(a.oracle_i,a.output,a.frames))
