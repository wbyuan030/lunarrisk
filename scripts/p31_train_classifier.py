#!/usr/bin/env python3
"""Mini-batch P3.1 classifier. Train/Val only; full benchmark scoring is a separate gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data","stats","class-map","outdir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--task",choices=("age","geology"),default="age")
    parser.add_argument("--track",choices=("official_stored","official_geographic"),default="official_stored")
    parser.add_argument("--mode",choices=("train","replay"),default="train")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--stop-after-epoch",type=int)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--purpose",choices=("runner_validation","baseline"),default="runner_validation")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--export-features",action="store_true")
    args=parser.parse_args(argv)
    if args.epochs<1 or args.batch_size<1 or args.workers<0 or args.lr<=0 or args.patience<0:
        raise ValueError("invalid training budget")
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<=args.epochs:
        raise ValueError("stop epoch must be within the unchanged planned horizon")
    if args.mode=="replay" and (args.checkpoint is None or args.resume is not None):
        raise ValueError("replay requires --checkpoint and forbids --resume")
    if args.mode=="train" and args.checkpoint is not None:
        raise ValueError("use --resume for continuation")
    import torch
    from lunarrisk.benchmarks.classifier import (
        ClassifierDataset,MetadataTargets,ScratchClassifier,make_loader,evaluate,seed_all,rng_state,restore_rng,
        cpu_state,state_fingerprint,atomic_checkpoint,learning_rate,stream_sha256,export_features,peak_rss_bytes)
    seed_all(args.seed)
    device=torch.device(args.device)
    if device.type=="cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested existing GPU is unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    args.outdir.mkdir(parents=True,exist_ok=False)
    hash_start=time.monotonic()
    # Once per invocation, never once per epoch; full content verification is not inferred from mtime.
    data_hash=stream_sha256(args.data)
    if args.expected_data_sha256 and data_hash!=args.expected_data_sha256:
        raise ValueError("full data SHA256 does not match expected asset")
    inputs={"data_sha256":data_hash,"data_bytes":args.data.stat().st_size,
            "stats_sha256":stream_sha256(args.stats),"class_map_sha256":stream_sha256(args.class_map)}
    asset={**inputs,"data_revision":"53099efd0f2db81eee96b28eae87e1efdd281310",
           "full_content_hash_verified_this_invocation":True,"hash_chunk_bytes":4*1024*1024,
           "hash_seconds":time.monotonic()-hash_start}
    (args.outdir/"asset_manifest.json").write_text(json.dumps(asset,indent=2)+"\n")
    stats,mapping=json.loads(args.stats.read_text()),json.loads(args.class_map.read_text())
    datasets={name:ClassifierDataset(args.data,stats,mapping,args.task,code,args.track)
              for name,code in (("train",0),("val",1))}
    if set(map(str,datasets["train"].ids[datasets["train"].indices])) & set(map(str,datasets["val"].ids[datasets["val"].indices])):
        raise ValueError("Train/Val ID overlap")
    splits={name:{"rows":len(ds),"ordered_ids_sha256":ds.ordered_ids_hash,"class_counts":ds.class_counts.tolist()}
            for name,ds in datasets.items()}
    config={"task":args.task,"track":args.track,"planned_epochs":args.epochs,"batch_size":args.batch_size,
            "workers":args.workers,"seed":args.seed,"lr":args.lr,"weight_decay":0.01,"grad_clip":1.0,
            "patience":args.patience,"precision":args.precision,"purpose":args.purpose,
            "schedule":"upstream cosine with min(2 epochs, total_steps//10) warmup",
            "selection":"strictly higher Val full-K macro-F1; earliest epoch on ties"}
    plan_hash=None
    if args.purpose=="baseline":
        if args.plan is None:
            raise ValueError("baseline requires a prelocked --plan")
        plan=json.loads(args.plan.read_text())
        plan_hash=stream_sha256(args.plan)
        if (data_hash!=plan["data"]["sha256"] or args.track!=plan["data"]["track"] or
                inputs["data_bytes"]!=plan["data"]["bytes"] or
                any(inputs[k]!=plan["data"][k] for k in ("stats_sha256","class_map_sha256"))):
            raise ValueError("baseline data/statistics/map/track differs from locked plan")
        for name in ("train","val"):
            if (len(datasets[name])!=plan["data"][name+"_rows"] or
                    datasets[name].ordered_ids_hash!=plan["data"][name+"_ordered_ids_sha256"]):
                raise ValueError("formal baseline split differs from locked IDs")
        for key in ("epochs","batch_size","seed","lr","patience","precision","workers"):
            allowed=plan["training"][key]
            value=getattr(args,key)
            mismatch = value not in allowed if isinstance(allowed,list) else value!=allowed
            if mismatch:
                raise ValueError("baseline configuration differs from locked plan: "+key)
    source_paths=(Path(__file__).resolve(),ROOT/"src/lunarrisk/benchmarks/classifier.py",ROOT/"src/lunarrisk/benchmarks/moonstone.py")
    sources={str(p.relative_to(ROOT)):stream_sha256(p) for p in source_paths}
    classes=5 if args.task=="age" else 49
    model=ScratchClassifier(classes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    generators={name:torch.Generator().manual_seed(args.seed+i) for i,name in enumerate(datasets)}
    loaders={name:make_loader(ds,args.batch_size,name=="train",generators[name],args.workers) for name,ds in datasets.items()}
    completed,best_metric,best_epoch,no_improve=0,-1.0,0,0
    history,best_state=[],None
    previous=args.resume if args.mode=="train" else args.checkpoint
    if previous:
        payload=torch.load(previous,map_location="cpu",weights_only=True)
        if (payload["config"]!=config or payload["inputs"]!=inputs or payload["splits"]!=splits or
                payload["source_sha256"]!=sources or payload["plan_sha256"]!=plan_hash):
            raise ValueError("checkpoint config/data/split/source/plan mismatch")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed=payload["completed_epochs"]
        best_metric,best_epoch,no_improve=payload["best_metric"],payload["best_epoch"],payload["no_improve"]
        history,best_state=payload["history"],payload["best_state_dict"]
        for name,generator in generators.items():
            generator.set_state(payload["loader_rng"][name])
        restore_rng(payload["rng"])
    begin=time.monotonic()
    train_seconds,train_examples=0.0,0
    if args.mode=="train":
        end=args.stop_after_epoch if args.stop_after_epoch is not None else args.epochs
        if completed>=end or (args.patience and no_improve>=args.patience):
            raise ValueError("checkpoint already completed/stopped under this locked budget")
        for epoch in range(completed,end):
            model.train()
            if device.type=="cuda": torch.cuda.synchronize(device)
            start=time.monotonic()
            total_loss,seen,max_batch=0.0,0,0
            for batch_index,batch in enumerate(loaders["train"]):
                x,y=batch["pixels"].to(device),batch["target"].to(device)
                lr=learning_rate(epoch*len(loaders["train"])+batch_index,args.epochs*len(loaders["train"]),len(loaders["train"]),args.lr)
                for group in optimizer.param_groups: group["lr"]=lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=args.precision=="bf16"):
                    loss=torch.nn.functional.cross_entropy(model(x),y)
                if not torch.isfinite(loss): raise ValueError("nonfinite loss")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                total_loss+=float(loss.detach())*len(y)
                seen+=len(y)
                max_batch=max(max_batch,len(y))
            if device.type=="cuda": torch.cuda.synchronize(device)
            elapsed=time.monotonic()-start
            train_seconds+=elapsed
            train_examples+=seen
            metrics=evaluate(model,loaders["val"],device,classes,precision=args.precision)
            if metrics["macro_f1_full"]>best_metric:
                best_metric,best_epoch,best_state,no_improve=metrics["macro_f1_full"],epoch+1,cpu_state(model),0
            else:
                no_improve+=1
            row={"epoch":epoch+1,"train_loss":total_loss/seen,"train_rows":seen,"largest_train_batch":max_batch,
                 "train_seconds":elapsed,"train_samples_per_second":seen/elapsed,"val":metrics}
            history.append(row)
            completed=epoch+1
            payload={"format_version":1,"architecture":"P31_scratch_CNN_28_64_128_256", "protocol":"HDF5_DERIVED_PILOT",
                     "code_sha":os.environ.get("LUNARRISK_GIT_SHA","UNKNOWN"),"source_sha256":sources,
                     "config":config,"inputs":inputs,"splits":splits,"plan_sha256":plan_hash,
                     "completed_epochs":completed,"state_dict":cpu_state(model),"optimizer_state_dict":optimizer.state_dict(),
                     "best_metric":best_metric,"best_epoch":best_epoch,"best_state_dict":best_state,"no_improve":no_improve,
                     "history":history,"rng":rng_state(),"loader_rng":{k:g.get_state() for k,g in generators.items()},
                     "scaler":{"enabled":False,"reason":"fp32 or bf16; no fp16 scaler"},"test_evaluated":False}
            atomic_checkpoint(args.outdir/"checkpoint.pt",payload)
            print(json.dumps(row),flush=True)
            if args.patience and no_improve>=args.patience: break
    last_fingerprint=state_fingerprint(payload["state_dict"])
    model.load_state_dict(best_state)
    final={}
    for name,ds in datasets.items():
        ordered=make_loader(ds,args.batch_size,False,torch.Generator().manual_seed(0),args.workers)
        final[name]=evaluate(model,ordered,device,classes,args.outdir/(name+"_predictions.csv"),name,args.precision)
    prior=datasets["train"].class_counts.astype(float)/len(datasets["train"])
    prior_metrics={name:evaluate(model,make_loader(MetadataTargets(ds),args.batch_size,False,
                          torch.Generator().manual_seed(0)),device,classes,
                          args.outdir/(name+"_prior_predictions.csv"),name,prior=prior)
                   for name,ds in datasets.items()}
    if args.export_features:
        export_features(model,datasets,args.outdir/"features.h5",device,args.batch_size,
                        {"inputs":inputs,"source_sha256":sources,"splits":splits,"config":config,
                         "plan_sha256":plan_hash,"selected_epoch":best_epoch})
    report={"status":"P31_RUNNER_VALIDATION" if args.purpose=="runner_validation" else "P31_BASELINE_TRAINVAL",
            "purpose":args.purpose,"mode":args.mode,"protocol":"HDF5_DERIVED_PILOT","test_evaluated":False,
            "code_sha":payload["code_sha"],"source_sha256":sources,"config":config,"inputs":inputs,"splits":splits,
            "completed_epochs":completed,"best_epoch":best_epoch,"selection":config["selection"],"metrics":final,
            "train_prior_probabilities":prior.tolist(),"train_prior_metrics":prior_metrics,
            "last_weights_sha256":last_fingerprint,"best_weights_sha256":state_fingerprint(best_state),
            "prediction_sha256":{name:stream_sha256(args.outdir/(name+"_predictions.csv")) for name in datasets},
            "peak_process_rss_bytes":peak_rss_bytes(),"peak_rss_scope":"parent process; workers=0 initially",
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
            "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,
            "train_window_seconds_this_process":train_seconds,"train_examples_this_process":train_examples,
            "train_samples_per_second_this_process":train_examples/train_seconds if train_seconds else None,
            "train_and_export_wall_seconds":time.monotonic()-begin,"asset_hash_seconds":asset["hash_seconds"],
            "torch":str(torch.__version__),"hip":torch.version.hip,"parameters":sum(p.numel() for p in model.parameters())}
    (args.outdir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for ds in datasets.values(): ds.close()
    print(json.dumps({k:v for k,v in report.items() if k!="metrics"},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
