import time
from chembl_webresource_client.new_client import new_client

print("연결 시도 중...")
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

print(f"쿼리 객체 생성: {time.time()-start:.2f}초")

# 처음 5개만 가져와서 확인
count = 0
for r in results:
    print(r)
    count += 1
    if count >= 5:
        break

print(f"5개 받는데 걸린 시간: {time.time()-start:.2f}초")
