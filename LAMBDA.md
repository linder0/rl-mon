# GPU training on Lambda Cloud

The repeatable flow for running `train_mjx.py` on a rented Lambda GPU.
Everything is wrapped by `scripts/lambda.sh`.

## One-time setup (already done)

- **API key** lives at `~/.config/lambda/api_key` (chmod 600). It is *not*
  in the repo and must never be committed. Rotate it in the
  [Lambda console](https://cloud.lambdalabs.com) if it leaks.
- **SSH key**: your local `~/.ssh/id_ed25519` is registered in Lambda as
  `linda-local`, so new instances accept it automatically.

## The flow

```bash
scripts/lambda.sh types                    # what has capacity right now, $/hr
scripts/lambda.sh launch                   # 1x A100 (default), waits for boot, prints IP
scripts/lambda.sh setup <ip>               # rsync repo, install requirements-mjx.txt,
                                           #   assert JAX sees the GPU, run mjx_envs.validate
scripts/lambda.sh train <ip> Ant-v5        # start training under nohup (disconnect-safe)
scripts/lambda.sh watch <ip>               # tail the training log (Ctrl-C to stop watching)
scripts/lambda.sh sync <ip>                # pull runs/ back to this machine
scripts/lambda.sh status                   # instance ids + IPs
scripts/lambda.sh terminate <instance-id>  # STOP PAYING — do this when done!
```

Then export locally as usual (the run folder is already in `runs/`):

```bash
python export_onnx.py --all-envs           # ONNX + parity + transfer check
cd web && npm run dev                      # watch it in the browser
```

Extra `train` args pass straight through to `train_mjx.py`:

```bash
scripts/lambda.sh train <ip> Ant-v5 --num-envs 8192 --seed 1
```

## Notes

- `train_mjx.py` pre-flights the GPU and refuses to run on CPU, so a broken
  CUDA setup fails loudly at `setup`/`train` rather than silently burning
  money at 1/1000th speed.
- Training runs under `nohup`; losing your connection doesn't kill the run.
  `watch` re-attaches to the log anytime.
- Budget: Ant-v5 (100M steps, 4096 envs) is ~tens of minutes on an A100
  ($1.99/hr) — expect the whole session to cost a dollar or two. The A10
  ($1.29/hr) also works, just slower.
- Instance disks vanish at terminate. Always `sync` before `terminate`.
- The Ant acceptance gate: after export, check the
  `transfer check (C-MuJoCo)` line — want ~1000 steps survived and clearly
  positive forward distance, then confirm visually in the web viewer.
