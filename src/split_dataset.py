import pandas as pd
from sklearn.model_selection import train_test_split

print("Dataset split 시작")

df = pd.read_csv(
    "data/processed/egfr_processed.csv"
)

print("전체:", df.shape)


# train 80%, temp 20%
train, temp = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# val/test 10%씩
val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42
)


print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)


train.to_csv(
    "data/processed/train.csv",
    index=False
)

val.to_csv(
    "data/processed/val.csv",
    index=False
)

test.to_csv(
    "data/processed/test.csv",
    index=False
)


print("저장 완료")

