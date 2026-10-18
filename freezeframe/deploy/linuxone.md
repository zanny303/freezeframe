# Running FreezeFrame on LinuxONE Community Cloud

These steps have not been run yet (no LinuxONE access from the dev machine). They are the
checklist for the first session on the VM; the Roadmap in README.md tracks what is outstanding.

## 1. VM

Request a VM at https://community.ibm.com/zsystems/form/l1cc-oss-vm-request/ (RHEL 9 or
Ubuntu 22.04, s390x). Note the public IP and download the SSH key.

```bash
ssh -i <key> linux1@<ip>
uname -m                       # s390x
nproc; free -g; df -h          # record cores, RAM, disk for the results slide
```

## 2. Hardware checks

```bash
# IBM Z Integrated Accelerator for AI (z16+): the NNPA facility shows up in cpu features
grep -o -w nnpa /proc/cpuinfo | head -1 || echo "no NNPA (AI accelerator not exposed)"
# Secure Execution guest?
cat /sys/firmware/uv/prot_virt_guest 2>/dev/null || echo "not a Secure Execution guest"
```

The service and benchmark report both: `GET /v1/enclave` returns `attestation`
("secure-execution" or "simulated"), and `bench/latency.py` records `scoring_tree_format`
("zdnn_tensors" means Snap ML is scoring on the AI accelerator, "compress_trees" means CPU).

## 3. Install

```bash
sudo dnf install -y python3.11 python3.11-pip git   # or apt install python3.11 python3.11-venv
git clone https://github.com/adipkg/freezeframe && cd freezeframe
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,xgb]'      # snapml has an s390x wheel; xgboost builds/installs on s390x
python -c "from snapml import GraphFeaturePreprocessor; print('ok')"
```

If `xgboost` has no s390x wheel for the pinned range, either build it (`pip install
xgboost --no-binary xgboost`, needs cmake + g++) or train on x86 and copy
`artifacts/models/` over: the bundles store XGBoost JSON and re-import it into Snap ML on load,
and `import_model(..., tree_format='auto')` selects the accelerator on the target machine.

## 4. Data

Download on the VM (no Kaggle login needed) or `scp` from a laptop:

```bash
mkdir -p data && for f in HI-Small_Trans.csv HI-Small_Patterns.txt; do
  curl -L -o data/$f https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/$f
done
```

## 5. Run

```bash
./scripts/run_experiments.sh HI-Small                 # ~1 h on an 8-core laptop
python -m bench.latency --speeds 1 100 1000 3000 --seconds 60
python -m demo.app                                    # then tunnel: ssh -L 8000:localhost:8000 ...
```

Copy `results/` back and commit it; the slide numbers come from those JSON files.

## 6. Production enclave (design, not built)

- Image: `deploy/Dockerfile`, built for s390x and signed.
- Runtime: IBM Confidential Computing Container Runtime (formerly Hyper Protect Virtual
  Servers) on Secure Execution. The contract's `workload` section runs the image; the `env`
  section, encrypted to the HPVS key, carries the pseudonym key seed, so no operator sees it.
- Attestation: the enclave publishes its public key together with the attestation record; each
  bank verifies the image measurement before sealing payments to that key.
