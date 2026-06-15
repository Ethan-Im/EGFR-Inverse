import time
from chembl_webresource_client.new_client import new_client
import pandas as pd

print("EGFR bioactivity 데이터 수집 시작...")
start = time.time()

activity = new_client.activity
results = activity.filter(
    target_chembl_id="CHEMBL203",
    standard_type__in=["IC50", "Ki"],
    standard_units="nM",
    pchembl_value__isnull=False,
).only([
    "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_value", "pchembl_value"
])

records = []
for i, r in enumerate(results):
    records.append(r)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}개 수집... ({time.time()-start:.1f}초 경과)")

df = pd.DataFrame(records)
print(f"\n전체 수집 완료: {df.shape}, 총 {time.time()-start:.1f}초")

df.to_csv("data/raw/egfr_bioactivity_raw.csv", index=False)
print("저장 완료: data/raw/egfr_bioactivity_raw.csv")
