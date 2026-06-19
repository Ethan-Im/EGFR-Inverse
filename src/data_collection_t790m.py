import time
from chembl_webresource_client.new_client import new_client
import pandas as pd

print("EGFR T790M 변이 데이터 수집 시작...")
start = time.time()

activity = new_client.activity
results = activity.filter(
    target_chembl_id="CHEMBL203",
    standard_type__in=["IC50", "Ki"],
    standard_units="nM",
    pchembl_value__isnull=False,
).only([
    "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_value", "pchembl_value",
    "assay_description", "assay_chembl_id"
])

records = []
for i, r in enumerate(results):
    desc = (r.get("assay_description") or "")
    if "T790M" in desc.upper():
        records.append(r)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}개 검사... ({time.time()-start:.0f}초 경과, T790M 발견: {len(records)}개)")

df = pd.DataFrame(records)
print(f"\n전체 수집 완료: {df.shape}, 총 {time.time()-start:.0f}초")

df.to_csv("data/raw/egfr_t790m_raw.csv", index=False)
print("저장 완료: data/raw/egfr_t790m_raw.csv")
