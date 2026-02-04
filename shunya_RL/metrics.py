import numpy as np

def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def prf(tp, tn, fp, fn):
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return acc, prec, rec, f1

def print_metrics(name, y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    acc, prec, rec, f1 = prf(tp, tn, fp, fn)
    print(f"\n[{name}]")
    print(f"  n={len(y_true)}  pos={int(np.sum(y_true))}")
    print(f"  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    print(f"  tp={tp}  tn={tn}  fp={fp}  fn={fn}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(y_true)}
