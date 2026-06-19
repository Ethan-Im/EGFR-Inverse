import pandas as pd
import re

df = pd.read_csv("data/raw/egfr_t790m_raw.csv")

def classify_mutation(desc):
    desc = str(desc).upper()
    has_t790m = "T790M" in desc
    has_l858r = "L858R" in desc
    has_c797s = "C797S" in desc
    has_del19 = "DEL19" in desc or "DEL 19" in desc or "DELETION" in desc and "19" in desc or "D746" in desc or "E746" in desc or "746" in desc

    if has_t790m and has_l858r and has_c797s:
        return "Triple_T790M_L858R_C797S"
    elif has_t790m and has_c797s:
        return "Double_T790M_C797S"
    elif has_t790m and has_l858r:
        return "Double_T790M_L858R"
    elif has_t790m and has_del19:
        return "Double_T790M_Del19"
    elif has_t790m:
        return "Single_T790M"
    else:
        return "Other"

df["mutation_category"] = df["assay_description"].apply(classify_mutation)

print("=== 변이 카테고리별 레코드 수 ===")
print(df["mutation_category"].value_counts())
print()

df.to_csv("data/raw/egfr_t790m_classified.csv", index=False)
print("저장 완료: data/raw/egfr_t790m_classified.csv")
