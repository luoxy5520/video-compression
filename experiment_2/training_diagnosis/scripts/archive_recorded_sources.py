"""Archive source bytes matching recorded evaluation hashes, using repository history.
This preserves evaluation provenance; it does not assert bitwise GPU-training replay.
"""
from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parents[1];repo=R.parents[1];archive=R/'source_snapshots'
def digest(data):return hashlib.sha256(data).hexdigest()
summary=[]
for snapshot in (R/'results').rglob('source_snapshot.json'):
    record=json.loads(snapshot.read_text());sources=record.get('scripts',{})
    if not sources:continue
    tag=digest(snapshot.read_bytes())[:12];out=archive/('recorded_'+tag);out.mkdir(parents=True,exist_ok=True);unmatched=[]
    for rel,expected in sources.items():
        name=Path(rel).name;p=R/'scripts'/name;candidates=[]
        if p.exists():candidates.append(p.read_bytes())
        candidates.extend(q.read_bytes() for q in archive.rglob(name))
        matched=next((b for b in candidates if digest(b)==expected),None)
        if matched is None:
            path=p.relative_to(repo).as_posix();history=subprocess.run(['git','log','--all','--format=%H','--',path],cwd=repo,capture_output=True,text=True,check=True).stdout.splitlines()
            for commit in history:
                result=subprocess.run(['git','show',commit+':'+path],cwd=repo,capture_output=True)
                if result.returncode==0 and digest(result.stdout)==expected:matched=result.stdout;break
        if matched is None:unmatched.append(rel);continue
        target=out/name
        if target.exists():assert target.read_bytes()==matched,'archive collision'
        else:target.write_bytes(matched)
    manifest=dict(snapshot=str(snapshot.relative_to(R)),snapshot_sha256=digest(snapshot.read_bytes()),expected_files=len(sources),matched_files=len(sources)-len(unmatched),unrecovered=unmatched,scope='exact recorded evaluation source hashes; restore original scripts layout to reproduce')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');summary.append(manifest)
print(json.dumps(summary,indent=2))
