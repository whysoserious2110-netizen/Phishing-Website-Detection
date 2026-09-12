import pandas as pd


print("===================================")
print("PREPARE MIXED TRAINING DATA V4")
print("===================================")


# ===================================
# 1. LOAD V3 DATA
# ===================================

v3 = pd.read_csv("mixed_training_data_v3.csv")

print("V3 rows:", len(v3))


# ===================================
# 2. LOAD V4 LEGITIMATE AUTH DATA
# ===================================

auth = pd.read_csv("legitimate_auth_urls_v4.csv")

print("Curated auth URLs:", len(auth))


# ===================================
# 3. KEEP ONLY NEW URLS
# ===================================

existing_urls = set(
    v3["URL"].astype(str)
)

new_auth = auth[
    ~auth["URL"].astype(str).isin(existing_urls)
].copy()

print("New URLs not already in V3:", len(new_auth))
print()


# ===================================
# 4. ADD SOURCE INFORMATION
# ===================================

new_auth["source"] = "Curated_legitimate_auth_v4"


# ===================================
# 5. COMBINE WITH V3
# ===================================

v4 = pd.concat(
    [
        v3,
        new_auth[["URL", "label", "source"]]
    ],
    ignore_index=True
)


# ===================================
# 6. REMOVE DUPLICATES
# ===================================

before = len(v4)

v4 = v4.drop_duplicates(
    subset=["URL"],
    keep="first"
)

duplicates_removed = before - len(v4)


# ===================================
# 7. SAVE
# ===================================

v4.to_csv(
    "mixed_training_data_v4.csv",
    index=False
)


# ===================================
# 8. SUMMARY
# ===================================

print("===================================")
print("V4 DATASET SUMMARY")
print("===================================")

print("V3 rows:", len(v3))
print("New legitimate auth rows:", len(new_auth))
print("Duplicates removed:", duplicates_removed)
print("Final V4 rows:", len(v4))

print()
print("Class distribution:")
print(v4["label"].value_counts())

print()
print("Label meaning:")
print("0 = Phishing")
print("1 = Legitimate")

print()
print("New URLs added:")

if len(new_auth) > 0:
    print(
        new_auth[["URL", "label"]]
        .to_string(index=False)
    )
else:
    print("None")

print()
print("Saved:")
print("mixed_training_data_v4.csv")