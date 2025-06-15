import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    gt_path = 'data/processed/submission.parquet'
    pred_path = 'output_A.csv'

    df_gt = pd.read_parquet(gt_path)
    df_pred = pd.read_csv(pred_path)

    key_cols = ['src_id', 'dst_id', 'edge_id', 'start_time', 'end_time']
    df = pd.merge(df_gt, df_pred, on=key_cols, how='left')

    df_eval = df.dropna(subset=['label', 'probability'])

    y_true = df_eval['label'].astype(int).values
    y_proba = df_eval['probability'].values

    # Compute metrics
    auc = roc_auc_score(y_true, y_proba)
    print(f"AUC      : {auc:.4f}")
    thresholds = np.linspace(0.0, 1.0, 10000)
    results = []
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_true, y_pred_thr)
        precision_thr = precision_score(y_true, y_pred_thr)
        recall_thr = recall_score(y_true, y_pred_thr)
        f1_thr = f1_score(y_true, y_pred_thr)
        results.append((thr, acc_thr, precision_thr, recall_thr, f1_thr))
    df_thr = pd.DataFrame(results, columns=['threshold','accuracy','precision','recall','f1'])
    best = df_thr.loc[df_thr['f1'].idxmax()]
    print(f"Best threshold: {best.threshold:.4f} -> Acc:{best.accuracy:.4f}, Prec:{best.precision:.4f}, Rec:{best.recall:.4f}, F1:{best.f1:.4f}")
    df_thr.to_csv('threshold_iteration.csv', index=False)

if __name__ == '__main__':
    main()
