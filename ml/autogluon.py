import pandas as pd
import mlflow
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score, average_precision_score,
    confusion_matrix, precision_score,)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl.styles import PatternFill

df = pd.read_csv("/Users/mar/BIO/PROJECTS/MPOX/COMMENT/APOBEC3-Mpox-Hairpins/ml/ml_table_all.tsv", sep='\t')
df = df[((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 7)) |
        ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 6)) |
        ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 5)) |
        ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 6)) |
        ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 5))]

feature_set1 = ["grantham",
                "third_nt",
                #"in_stem","stem_length","stem_loop_length","stem_energy",
               "in_loop_boundary","loop_boundary_length","loop_boundary_stem_length","loop_boundary_energy"#,
               #"in_loop_other","loop_other_length","loop_other_stem_length","loop_other_energy"
               ]

target = "any_mutation"
preset = "medium"
metric = "roc_auc"

def log_autogluon(predictor, test_df=None):
    lb = predictor.leaderboard()
    for _, r in lb.iterrows():
        mlflow.log_metric(f"score_val__{r['model'].replace('.', '_')}", r["score_val"])
        #mlflow.log_metric(f"roc_auc_val__{r['model'].replace('.', '_')}", r["score_val_roc_auc"])
    mlflow.log_metric(f"best_score_val", lb.iloc[0]["score_val"])
    #mlflow.log_metric(f"best_rocauc_val", lb.iloc[0]["score_val_roc_auc"])
    mlflow.log_param("best_model", lb.iloc[0]["model"])
    lb.to_csv("leaderboard.csv", index=False)
    mlflow.log_artifact("leaderboard.csv")

def save_predictions_xlsx(out, target, path="predictions.xlsx"):
    # добавим явную колонку типа ошибки — удобно фильтровать в Excel
    def err_type(r):
        if r["pred_class"] == 1 and r[target] == 0: return "FP"
        if r["pred_class"] == 0 and r[target] == 1: return "FN"
        return ""
    out = out.copy()
    out["error"] = out.apply(err_type, axis=1)

    out.to_excel(path, index=False)

    # подсветка через openpyxl
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb.active

    fp_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # красный
    fn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # жёлтый

    err_col = out.columns.get_loc("error") + 1   # openpyxl считает с 1
    for row in range(2, ws.max_row + 1):          # строка 1 — заголовок
        val = ws.cell(row=row, column=err_col).value
        fill = fp_fill if val == "FP" else fn_fill if val == "FN" else None
        if fill:
            for col in range(1, ws.max_column + 1):
                ws.cell(row=row, column=col).fill = fill

    wb.save(path)
    return path


mlflow.set_experiment("autogluon-ds2-s7l3")

with mlflow.start_run():
    mlflow.log_params({
        "target": target,
        "eval_metric": metric,
        "presets": preset,
        "n_features": len(feature_set1),
        "n_rows": len(df),
    })
    mlflow.log_param("features", ",".join(feature_set1))

    predictor = TabularPredictor(label=target, eval_metric=metric, problem_type='binary', positive_class=1).fit(df[feature_set1 + [target]], presets=preset)
    #predictor = TabularPredictor(label=target).fit(df[feature_set1 + [target]], presets=preset)

    log_autogluon(predictor)

    y_true = df[target].values
    prob = predictor.predict_proba(df).iloc[:, -1].values

    threshold = prob[y_true == 1].min()
    y_pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    mlflow.log_params({"threshold": float(threshold)})
    mlflow.log_metrics({
        "recall": recall, "precision": precision,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "n_positives": int(tp + fn),
    })
                       
    odf = df.copy() 
    odf["pred_class"] = y_pred
    odf["pred_prob"] = prob
 
    run = mlflow.active_run()
    run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])

    odf.to_csv(f"predictions_{run_name}.csv", index=False)
    mlflow.log_artifact(f"predictions_{run_name}.csv")
    save_predictions_xlsx(odf, target, f"predictions_{run_name}.xlsx")

    prec, rec, _ = precision_recall_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR (AP = {pr_auc:.3f})")
    ax.scatter([recall], [precision], color="red", zorder=5,
               label=f"порог={threshold:.3f}\nrecall={recall:.2f}, prec={precision:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR-кривая (рабочая точка = recall 100%)")
    ax.legend(loc="lower left")
    mlflow.log_figure(fig, "pr_curve.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, prob)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC")
    ax2.legend(loc="lower right")
    mlflow.log_figure(fig2, "roc_curve.png")
    plt.close(fig2)



