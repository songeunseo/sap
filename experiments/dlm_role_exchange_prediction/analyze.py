#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.dlm_role_exchange_prediction.core import ROOT, RUNTIME, atomic_json, cluster_bootstrap_difference, ridge_fit_predict

TYPES=("attn_out","ff_out","ff_proj","k_proj","q_proj","up_proj","v_proj")
MODELS=("P0_structure","P1_pooled_dense","P2_role_dense","P3_role_dense_sparse","P4_role_energy")
SEEDS=(101,202,303)


def controls(row):
    return [row["layer"]/31,*(1.0 if row["projection_type"]==x else 0.0 for x in TYPES),
            row["base_level"]/5,1.0 if row["new_level"]>row["base_level"] else -1.0,
            np.sign(row["parameter_delta"])*np.log1p(abs(row["parameter_delta"]))]


def features(row,model,random_seed=None):
    result=controls(row); dense=row["features"]["dense"]; sparse=row["features"]["sparse"]
    prefix=f"random_{random_seed}_" if random_seed else ""
    if model=="P0_structure": return result
    if model=="P1_pooled_dense": return result+[dense["pooled_token"]]
    result += [dense[f"{prefix}masked_token"],dense[f"{prefix}unmasked_token"]]
    if model in ("P3_role_dense_sparse","P4_role_energy"):
        result += [sparse[f"{prefix}masked_token"],sparse[f"{prefix}unmasked_token"]]
    if model=="P4_role_energy":
        result += [dense[f"{prefix}masked_energy"],dense[f"{prefix}unmasked_energy"],
                   sparse[f"{prefix}masked_energy"],sparse[f"{prefix}unmasked_energy"]]
    return result


def flatten(payload):
    rows=[]; bundles=[]
    for state in payload["states"]:
        for row in state["rows"]:
            rows.append({**row,"document":state["sequence_index"],"timestep":state["timestep"]})
        for row in state["bundle_rows"]:
            bundles.append({**row,"document":state["sequence_index"],"timestep":state["timestep"]})
    return rows,bundles


def metrics(y,pred):
    y=np.asarray(y); pred=np.asarray(pred); error=(pred-y)**2
    return {"mse":float(error.mean()),"spearman":float(spearmanr(y,pred).statistic),
            "sign_accuracy":float(np.mean(np.sign(y)==np.sign(pred))),
            "calibration_slope":float(np.linalg.lstsq(np.column_stack([np.ones(len(pred)),pred]),y,rcond=None)[0][1]),
            "squared_error":error.tolist()}


def main():
    design=json.loads((ROOT/"design.json").read_text())
    dev,bdev=flatten(torch.load(RUNTIME/"development.pt",map_location="cpu",weights_only=False))
    final,bfinal=flatten(torch.load(RUNTIME/"final.pt",map_location="cpu",weights_only=False))
    predictions={name:np.full(len(final),np.nan) for name in MODELS}; random_predictions={seed:np.full(len(final),np.nan) for seed in SEEDS}
    y=np.asarray([x["delta_kl"] for x in final])
    for fold,layers in enumerate(design["layer_folds"]):
        train=[x for x in dev if x["layer"] not in layers]; test_indices=[i for i,x in enumerate(final) if x["layer"] in layers]
        test=[final[i] for i in test_indices]; yt=[x["delta_kl"] for x in train]
        for model in MODELS:
            pred=ridge_fit_predict([features(x,model) for x in train],yt,[features(x,model) for x in test],design["ridge_alpha"])
            predictions[model][test_indices]=pred
        for seed in SEEDS:
            pred=ridge_fit_predict([features(x,"P2_role_dense",seed) for x in train],yt,
                                   [features(x,"P2_role_dense",seed) for x in test],design["ridge_alpha"])
            random_predictions[seed][test_indices]=pred
    if any(np.isnan(x).any() for x in [*predictions.values(),*random_predictions.values()]): raise RuntimeError("OOF prediction missing")
    results={name:metrics(y,pred) for name,pred in predictions.items()}
    random_metrics={str(seed):metrics(y,pred) for seed,pred in random_predictions.items()}
    random_error=np.mean([(random_predictions[s]-y)**2 for s in SEEDS],axis=0)
    errors={name:(pred-y)**2 for name,pred in predictions.items()}
    docs=np.asarray([x["document"] for x in final]); quartiles=np.asarray([x["layer"]//8 for x in final])
    comparisons={}
    pairs=[("H1_P2_minus_P1","P2_role_dense","P1_pooled_dense"),
           ("H2_P3_minus_P2","P3_role_dense_sparse","P2_role_dense")]
    for label,left,right in pairs:
        comparisons[label]=cluster_bootstrap_difference(errors[left],errors[right],docs,quartiles)
        comparisons[label]["relative_mse_reduction"]=float((results[right]["mse"]-results[left]["mse"])/results[right]["mse"])
    comparisons["role_minus_random"]=cluster_bootstrap_difference(errors["P2_role_dense"],random_error,docs,quartiles)
    comparisons["role_minus_random"]["relative_mse_reduction"]=float((random_error.mean()-results["P2_role_dense"]["mse"])/random_error.mean())

    pred_lookup={(row["document"],row["timestep"],row["exchange_index"]):{m:float(predictions[m][i]) for m in MODELS}
                 for i,row in enumerate(final)}
    actual_lookup={(row["document"],row["timestep"],row["exchange_index"]):row["delta_kl"] for row in final}
    bundle_results={}
    for model in MODELS:
        actual=[]; predicted=[]; additive=[]; bdocs=[]; bq=[]
        for row in bfinal:
            key=lambda e:(row["document"],row["timestep"],e)
            actual.append(row["delta_kl"]); predicted.append(sum(pred_lookup[key(e)][model] for e in row["exchange_indices"]))
            additive.append(sum(actual_lookup[key(e)] for e in row["exchange_indices"])); bdocs.append(row["document"]); bq.append(row["layer_quartile"])
        bundle_results[model]={**metrics(actual,predicted),"mean_nonadditive_effect":float(np.mean(np.asarray(actual)-np.asarray(additive))),
                               "mean_individual_prediction_error":float(np.mean(np.asarray(additive)-np.asarray(predicted)))}

    both_better_bad=0; both_better_total=0
    for row in final:
        d=row["features"]["sparse"]
        if d["masked_token"]<=0 and d["unmasked_token"]<=0:
            both_better_total+=1; both_better_bad += row["delta_kl"]>0
    gate={"H1_pass":comparisons["H1_P2_minus_P1"]["bootstrap_ci"][1]<0 and comparisons["H1_P2_minus_P1"]["relative_mse_reduction"]>=.10,
          "H2_pass":comparisons["H2_P3_minus_P2"]["bootstrap_ci"][1]<0 and comparisons["H2_P3_minus_P2"]["relative_mse_reduction"]>=.10,
          "role_control_pass":comparisons["role_minus_random"]["bootstrap_ci"][1]<0 and comparisons["role_minus_random"]["relative_mse_reduction"]>=.10}
    output={"status":"complete","models":results,"random_role_models":random_metrics,"comparisons":comparisons,
            "bundle_results":bundle_results,"both_role_reconstruction_better_but_kl_worse":{"count":int(both_better_bad),"total":both_better_total,
            "fraction":float(both_better_bad/both_better_total) if both_better_total else None},"gate":gate,
            "counts":{"development_rows":len(dev),"final_rows":len(final),"final_bundles":len(bfinal)}}
    atomic_json(ROOT/"analysis.json",output)
    lines=["# Role Exchange Prediction Validation","","## Results","",f"Rows: dev {len(dev)}, final {len(final)}; bundle observations {len(bfinal)}.","",
           "| Model | MSE | Spearman | Sign accuracy |","|---|---:|---:|---:|"]
    for name in MODELS: lines.append(f"| {name} | {results[name]['mse']:.8g} | {results[name]['spearman']:.4f} | {results[name]['sign_accuracy']:.4f} |")
    lines += ["","## Primary comparisons",""]
    for name,value in comparisons.items(): lines.append(f"- {name}: error difference {value['mean_error_difference']:.8g}, simultaneous CI {value['bootstrap_ci']}, relative reduction {value['relative_mse_reduction']:.2%}.")
    lines += ["","## Gate","",json.dumps(gate,sort_keys=True),"",f"Both role reconstruction improved but KL worsened: {both_better_bad}/{both_better_total}."]
    (ROOT/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"event":"analysis_complete","gate":gate,"comparisons":comparisons},sort_keys=True))

if __name__=="__main__":main()
