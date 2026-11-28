"""Read only the60 declared unsealed train/development files; test records excluded."""
from common import *
records=[r for r in json.loads((E2/'dataset_manifest.json').read_text())['records'] if r['role'] in ('train_candidate','historical_development')]
rows=[]
for r in records:
    x=tensor(r);assert list(x.shape)==[5,16,60,104] and torch.isfinite(x).all()
    rows.append(dict(path=r['path'],role=r['role'],group=r['group'],source=r['source'],file_sha256=sha(r['path']),tensor_sha256_TCHW=hashlib.sha256(x.numpy().tobytes()).hexdigest(),shape=list(x.shape),dtype=str(x.dtype)))
train=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='historical_development'];group_overlap=sorted({r['group'] for r in train}&{r['group'] for r in dev});hash_overlap=sorted({r['tensor_sha256_TCHW'] for r in train}&{r['tensor_sha256_TCHW'] for r in dev})
save(ROOT/'results/dataset_integrity_audit.json',dict(train_videos=len(train),historical_development_videos=len(dev),unique_train_tensor_hashes=len({r['tensor_sha256_TCHW'] for r in train}),unique_development_tensor_hashes=len({r['tensor_sha256_TCHW'] for r in dev}),prompt_seed_group_overlap=group_overlap,identical_tensor_train_development_overlap=hash_overlap,sealed_test='NOT OPENED; only explicit train/development records were read',records=rows))
assert not group_overlap and not hash_overlap,'data leakage: do not certify development comparisons'
print('dataset integrity PASS',len(train),len(dev))
