#!/usr/bin/env python3
"""Persistent hybrid TPE tuning for the EditKG reproduction study.

Six deterministic seed configurations initialise each study (existing logs are
reused when available), followed by four TPE proposals. KGAT uses six plus two.
Every screening run sees only data/editkg-validation; the true test is exposed
only to the final retraining command.
"""
import argparse, json, os, re, shlex, subprocess
from pathlib import Path

import optuna

DATASETS = ("last-fm", "amazon-book", "yelp2018")
MODELS = ("bpr", "gcmc", "lightgcn", "sgl", "cke", "ktup", "kgcn",
          "kgat", "kgin", "kgcl", "kgrec", "krdn", "editkg")
SAFE = {"bpr", "gcmc", "lightgcn", "sgl", "cke", "ktup", "kgcn"}

SPACES = {
 "bpr":{"dim":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3,3e-3,5e-3,1e-2],"batch_size":[1024,2048,4096],"l2":[1e-5,1e-4,5e-4,1e-3,2.5e-3]},
 "gcmc":{"embedding_size":[64,75,128],"gcn_output_dim":[128,256,500],"dropout_prob":[0.1,0.3,0.5],"learning_rate":[3e-4,5e-4,1e-3,1e-2],"train_batch_size":[1024,2048,4096],"num_basis_functions":[2,4,8]},
 "lightgcn":{"dim":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3,3e-3],"batch_size":[1024,2048,4096],"layers":[1,2,3,4],"l2":[1e-5,1e-4,1e-3]},
 "sgl":{"dim":[64,128],"lr":[3e-4,5e-4,1e-3,3e-3],"batch_size":[1024,2048,4096],"layers":[2,3],"l2":[1e-5,1e-4],"dropout":[.05,.1,.2],"temperature":[.1,.2,.5],"ssl_weight":[.05,.1,.2]},
 "cke":{"embedding_size":[64,128],"kg_embedding_size":[64,128],"learning_rate":[1e-4,3e-4,5e-4,1e-3,3e-3],"train_batch_size":[1024,2048,4096],"reg":[1e-5,1e-4,1e-3,1e-2]},
 "ktup":{"embedding_size":[64,128],"learning_rate":[1e-4,3e-4,5e-4,1e-3,3e-3],"train_batch_size":[1024,2048,4096],"margin":[.5,1.,2.],"kg_weight":[.1,.5,1.],"align_weight":[.1,.5,1.]},
 "kgcn":{"embedding_size":[64,128],"learning_rate":[1e-4,3e-4,5e-4,1e-3],"train_batch_size":[1024,2048,4096],"n_iter":[1,2],"neighbor_sample_size":[4,8,16],"aggregator":["sum","concat","neighbor"],"reg_weight":[1e-7,1e-6,1e-5]},
 "kgat":{"embed_size":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3],"layer_size":["[64]","[64,64]","[128,128]"],"dropout":[0.,.1,.2],"alg_type":["gcn","graphsage","bi"],"reg":[1e-5,1e-4,1e-3]},
 "kgin":{"dim":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3],"context_hops":[1,2,3],"n_factors":[2,4,8],"sim_regularity":[1e-5,1e-4,1e-3],"mess_dropout_rate":[0.,.1,.2]},
 "kgcl":{"dim":[64,128],"lr":[1e-4,3e-4,1e-3],"mu":[.3,.5,.7,.9],"tau":[.1,.2,.5],"cl_weight":[.01,.05,.1,.2]},
 "kgrec":{"dim":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3],"mae_coef":[.05,.1,.2],"mae_msize":[128,256,512],"cl_coef":[.005,.01,.05],"cl_tau":[.2,.5,1.],"cl_drop_ratio":[.1,.3,.5]},
 "krdn":{"dim":[64,128],"lr":[1e-4,3e-4,5e-4,1e-3],"context_hops":[1,2,3],"gamma":[.1,.3,.5,.7],"margin":[.2,.5,.7,1.],"num_neg_sample":[100,200,400]},
 "editkg":{"dim":[64,100,128],"lr":[1e-4,3e-4,5e-4,1e-3],"context_hops":[1,2,3],"node_dropout_rate":[.3,.5,.75,1.],"margin":[.1,.2,.5,.7],"num_neg_sample":[1,2,4],"l2":[1e-5,1e-4,1e-3]},
}

def params_for_trial(trial, model):
    return {k: trial.suggest_categorical(k, values) for k, values in SPACES[model].items()}

def seed_params(model, index):
    return {k: values[(index + pos) % len(values)] for pos, (k, values) in enumerate(SPACES[model].items())}

def registered_seed_params(root, model, index):
    """Return the exact parameters behind an existing screening log."""
    if model not in SAFE:
        return seed_params(model, index)
    rows=[]
    for line in (root/"configs/tuning_candidates.tsv").read_text().splitlines()[1:]:
        name,_,argline=line.split("\t",2)
        if name==model: rows.append(argline)
    if index >= len(rows): return seed_params(model,index)
    tokens=shlex.split(rows[index]); raw={}; pos=0
    while pos < len(tokens):
        token=tokens[pos]
        if "=" in token: key,value=token[2:].split("=",1); pos+=1
        else: key=token[2:]; value=tokens[pos+1]; pos+=2
        raw[key]=value
    parsed={}
    for key,values in SPACES[model].items():
        source="reg_weights" if key=="reg" else key
        value=raw.get(source)
        if value is None: parsed[key]=values[0]
        elif key=="reg": parsed[key]=float(value.strip("[]").split(",")[0])
        elif isinstance(values[0],int): parsed[key]=int(value)
        elif isinstance(values[0],float): parsed[key]=float(value)
        else: parsed[key]=value
    return parsed

def cli(model, p):
    out=[]
    for key,value in p.items():
        if model == "cke" and key == "reg": out.append(f"--reg_weights=[{value},{value}]")
        elif model == "kgat" and key == "dropout": out += ["--node_dropout",f"[{value}]","--mess_dropout",f"[{value},{value}]"]
        elif model == "kgat" and key == "reg": out += ["--regs",f"[{value},{value}]"]
        elif model == "kgrec" and key == "mae_coef": out += ["--mae","--mae_coef",str(value)]
        elif key not in {"num_basis_functions"}: out += ["--"+key, str(value)]
        else: out += ["--num_basis_functions",str(value)]
    return out

def result_from(path):
    text=path.read_text(errors="replace") if path.exists() else ""
    matches=re.findall(r'VALIDATION_RESULT=(\{[^\n]+\})', text)
    if matches:
        value=json.loads(matches[-1]); return float(value["recall@20"]),float(value["ndcg@20"])
    # KGAT and legacy tables fallback.
    matches=re.findall(r'recall=\[([0-9.eE+-]+)[^\]]*\].*?ndcg=\[([0-9.eE+-]+)',text)
    if matches: return tuple(map(float,matches[-1]))
    raise RuntimeError(f"no validation metrics in {path}")

def command(root, model, dataset, params, epochs, validation=True, final_once=False):
    env=os.environ.copy(); env.update(GPU_ID=os.environ.get("GPU_ID","0"),
      EDITKG_EVAL_STEP=(str(epochs) if final_once else "5"),
      EDITKG_STOPPING_STEP=("10" if epochs > 25 else "4"))
    if validation: env["EDITKG_DATA_ROOT"]=str(root/"data/editkg-validation")
    cmd=[str(root/"scripts"/"run_baseline.sh"),model,dataset]+cli(model,params)
    if model in {"bpr","lightgcn","sgl"}:
        cmd += ["--epochs",str(epochs),"--patience",("10" if epochs>25 else "4")]
        if validation: cmd += ["--validation_only"]
    elif model in {"gcmc","cke","ktup","kgcn"}:
        # RecBole trials use its native separated validation runner.
        if validation:
            cmd=[str(root/".envs/editkg/bin/python"),str(root/"scripts"/"run_recbole_validation.py"),
                 "--model",model.upper(),"--dataset",dataset,"--config_files",str(root/"configs/recbole_lastfm.yaml")]+cli(model,params)
        cmd += [f"--epochs={epochs}","--eval_step=5",f"--stopping_step={10 if epochs>25 else 4}"]
    else:
        cmd += ["--epoch",str(epochs)]
        if model == "kgat": cmd += ["--verbose","5"]
    return cmd,env

def existing_seed(root, dataset, model, index):
    if model not in SAFE: return None
    registry=[]
    for line in (root/"configs/tuning_candidates.tsv").read_text().splitlines()[1:]:
        m,tag,_=line.split("\t",2)
        if m==model: registry.append(tag)
    if index >= len(registry): return None
    log=root/"results/runs/paper_hparams_tuning/tuning"/f"tune_{dataset}_{model}_{registry[index]}.log"
    try: return result_from(log)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu-id",default="0"); args=ap.parse_args()
    root=Path(__file__).resolve().parent.parent; os.environ["GPU_ID"]=args.gpu_id
    run=root/"results/runs/hybrid_tpe"; logs=run/"logs"; logs.mkdir(parents=True,exist_ok=True)
    storage=f"sqlite:///{run/'studies.sqlite3'}"
    selections=[]
    for dataset in DATASETS:
      for model in MODELS:
        total=8 if model=="kgat" else 10
        study=optuna.create_study(study_name=f"{dataset}-{model}",storage=storage,
          load_if_exists=True,direction="maximize",sampler=optuna.samplers.TPESampler(seed=2024,multivariate=True))
        if not study.trials:
            for i in range(6): study.enqueue_trial(registered_seed_params(root,model,i),user_attrs={"seed_index":i})
        def objective(trial):
            p=params_for_trial(trial,model); idx=trial.user_attrs.get("seed_index")
            reused=existing_seed(root,dataset,model,idx) if idx is not None else None
            if reused:
                recall,ndcg=reused; trial.set_user_attr("source","existing_screening")
            else:
                log=logs/f"{dataset}_{model}_{trial.number:03d}.log"
                cmd,env=command(root,model,dataset,p,25,validation=True)
                with log.open("w") as stream:
                    cp=subprocess.run(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if cp.returncode: raise RuntimeError(f"exit {cp.returncode}: {log}")
                recall,ndcg=result_from(log); trial.set_user_attr("source_log",str(log))
            trial.set_user_attr("ndcg@20",ndcg); return recall
        remaining=max(0,total-len([t for t in study.trials if t.state.is_finished()]))
        if remaining: study.optimize(objective,n_trials=remaining,catch=(RuntimeError,))
        completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        if not completed: continue
        best=max(completed,key=lambda t:(t.value,t.user_attrs.get("ndcg@20",-1)))
        selections.append({"dataset":dataset,"model":model,"params":best.params,
          "valid_recall":best.value,"valid_ndcg":best.user_attrs.get("ndcg@20"),"trial":best.number})
    (run/"selected.json").write_text(json.dumps(selections,indent=2,sort_keys=True)+"\n")
    # Validation-stage retraining (patience 10), followed by one true-test run.
    for choice in selections:
        dataset,model,p=choice["dataset"],choice["model"],choice["params"]
        v_log=logs/f"final_validation_{dataset}_{model}.log"
        cmd,env=command(root,model,dataset,p,150,validation=True)
        with v_log.open("w") as stream: cp=subprocess.run(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if cp.returncode: continue
        epochs=re.findall(r'early stopping at\s+(\d+)',v_log.read_text(errors="replace"))
        best_epoch=min(150,int(epochs[-1])+1) if epochs else 150
        t_log=logs/f"final_test_{dataset}_{model}.log"
        cmd,env=command(root,model,dataset,p,best_epoch,validation=False,final_once=True)
        with t_log.open("w") as stream: subprocess.run(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)

if __name__=="__main__": main()
