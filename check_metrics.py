
def check_consistency(name, acc, p, r, f1, n0=3632, n1=5760):
    n = n0 + n1
    print(f"--- {name} ---")
    print(f"Reported: Acc={acc}, P={p}, R={r}, F1={f1}")
    
    # 1. Check F1 consistency
    if p + r == 0:
        calc_f1 = 0
    else:
        calc_f1 = 2 * p * r / (p + r)
    
    diff_f1 = abs(calc_f1 - f1)
    print(f"Calculated F1: {calc_f1:.2f} (Diff: {diff_f1:.2f})")
    
    # 2. Derive Confusion Matrix from P and R (and N1)
    # TP = R * N1
    tp = (r / 100.0) * n1
    
    # P = TP / (TP + FP) => FP = TP * (1/P - 1)
    if p == 0:
        fp = 0 # specific case, though usually P!=0 if TP!=0
    else:
        fp = tp * ((100.0 / p) - 1)
        
    # FN = N1 - TP
    fn = n1 - tp
    
    # TN = N0 - FP (Assuming FP is correct for the dataset)
    tn = n0 - fp
    
    print(f"Derived Counts: TP={tp:.1f}, FP={fp:.1f}, TN={tn:.1f}, FN={fn:.1f}")
    
    # 3. Check consistency with Accuracy
    # Acc = (TP + TN) / N
    calc_acc = (tp + tn) / n * 100.0
    diff_acc = abs(calc_acc - acc)
    
    print(f"Calculated Acc (based on P, R, N0, N1): {calc_acc:.2f} (Diff: {diff_acc:.2f})")
    
    is_f1_ok = diff_f1 < 0.1
    is_acc_ok = diff_acc < 0.1
    
    if tn < 0:
        print("WARNING: Implied TN is negative! (Precision/Recall combination impossible for this N0)")
    
    if is_f1_ok and is_acc_ok:
        print("RESULT: Consistent")
    else:
        print("RESULT: Inconsistent")
    print("")

n0 = 3632
n1 = 5760

# Table 1: Model Accuracy Precision Recall F1-score
data1 = [
    ("CCRNet", 93.98, 93.98, 96.36, 95.16),
    ("r.m. LInfoNCE", 73.11, 73.84, 86.97, 79.87),
    ("r.m. LConsistency", 88.94, 90.79, 91.22, 91.00),
    ("r.m. LListNet", 83.94, 95.70, 77.29, 85.52),
    ("r.p. C-Enc", 67.57, 75.61, 69.57, 72.46),
    ("r.p. LSeq-ResNet", 69.98, 73.81, 79.15, 76.39)
]

# Table 2: Model Accuracy F1-score Precision Recall
# Note: Remapped to (Name, Acc, P, R, F1)
data2 = [
    ("ResNet [22]", 70.81, 77.61, 73.66, 75.58),
    ("SE-CapsNet [23]", 66.13, 65.88, 92.89, 77.09),
    ("Km-CapsNet [24]", 84.26, 87.06, 87.31, 87.18),
    ("MANDO-HGT [25]", 81.82, 88.50, 80.86, 84.51),
    ("DR-GCN [26]", 81.09, 80.20, 91.83, 85.62),
    ("TMP [26]", 81.88, 81.96, 90.34, 85.95),
    ("Peculiar [27]", 86.53, 87.02, 91.72, 89.31),
    # CCRNet (ours) is duplicate, skipping or checking again
    ("CCRNet (ours)", 93.98, 93.98, 96.36, 95.16)
]

print("Checking Table 1 Data:")
for d in data1:
    check_consistency(d[0], d[1], d[2], d[3], d[4], n0, n1)

print("Checking Table 2 Data:")
for d in data2:
    check_consistency(d[0], d[1], d[2], d[3], d[4], n0, n1)
