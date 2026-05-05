# Plateau-Cliff Robustness Experiment

This project is a clean experiment scaffold for testing Plateau-Cliff local norm shaping together with the Cross-Lipschitz robust training method from Hein and Andriushchenko (2017).

## Final Experiment Output

The experiment is designed to answer one question:

Can the proposed Plateau-Cliff local norm shaping improve robustness when it is trained jointly with Cross-Lipschitz regularization, while keeping the same clean-accuracy entry condition?

The final comparison has three dimensions.

### 1. Model and Method Dimension

The default full run compares these rows:

| Row name | Meaning |
| --- | --- |
| `Mamba_Base` | Mamba classifier trained only with cross-entropy until the shared clean validation accuracy target is reached. |
| `DNN_Base` | Fully connected baseline trained only with cross-entropy until the same target is reached. |
| `nmODE_Base` | Neural ODE baseline trained only with cross-entropy until the same target is reached. |
| `Mamba_CLR_0.005` | Mamba trained with Cross-Lipschitz regularization. This is the Hein-Andriushchenko baseline method. |
| `Mamba_CLR_PCP_Joint_0.005` | Mamba trained with Cross-Lipschitz regularization and the proposed Plateau-Cliff local norm shaping term in the same objective. This is the new method. |

By default, `config.json` sets `target_acc` to `0.97` and `stop_at_target` to `true`. This means each model is stopped at the same clean validation accuracy threshold before robustness testing. The goal is to avoid comparing a robust method trained to one clean accuracy against a baseline trained to a very different clean accuracy.

The current innovation is implemented only on top of the Mamba Cross-Lip setting because it is meant to modify the Cross-Lip robust-training objective itself. The other base architectures are included to show how ordinary clean-trained models behave under the same attack suite.

### 2. Robustness Evaluation Dimension

Each model row is evaluated under six perturbation or attack families:

| Attack family | Type | Purpose |
| --- | --- | --- |
| `Blur` | Non-adversarial corruption | Tests stability under Gaussian blur. |
| `Resize` | Non-adversarial corruption | Tests information loss from downsampling and upsampling. |
| `Noise` | Non-adversarial corruption | Tests stability under random pixel noise. |
| `PGD` | White-box adversarial attack | Standard iterative gradient attack. |
| `AutoPGD` | Stronger white-box adversarial attack | Multi-step or multi-restart PGD-style attack. |
| `SquareAttack` | Black-box adversarial attack | Query-style square patch perturbation attack. |

The full experiment uses three strengths per attack:

| Strength | Meaning |
| --- | --- |
| `Weak` | Mild perturbation or attack budget. |
| `Medium` | Moderate perturbation or attack budget. |
| `Extreme` | Severe perturbation or attack budget. |

The server chain test uses only one `Smoke` strength per attack so that it can verify the pipeline quickly.

### 3. Metric Dimension

The primary metric is post-attack classification accuracy:

```text
accuracy = number of correctly classified perturbed examples / number of evaluated examples
```

Higher is better. The reported number answers: after applying this specific perturbation or adversarial attack, what fraction of examples still receive the correct label?

The clean validation accuracy is used as a training gate, not as the main robustness result. The intended comparison is therefore:

```text
same clean-accuracy threshold -> different attacks -> compare remaining accuracy
```

## Result Files

Each run writes a timestamped group of files under `results/`.

| File pattern | Content |
| --- | --- |
| `*_summary.json` | Nested result dictionary: model -> attack -> strength -> accuracy. This is the main raw result file. |
| `*_summary.csv` | Flat table version of the same results, useful for paper tables or external plotting. |
| `*_heatmap.png` | Heatmap visualization. Rows are models/methods, columns are attack-strength pairs, cell values are accuracies. |
| `*_config.json` | Fully resolved runtime configuration used by the run. |
| `*_input_config.json` | Copy of the user-facing input config for reproducibility. |

The heatmap has this layout:

```text
rows    = Mamba_Base, DNN_Base, nmODE_Base, Mamba_CLR_0.005, Mamba_CLR_PCP_Joint_0.005
columns = B-W, B-M, B-E, R-W, ..., SQ-E
values  = post-attack accuracy
```

Column abbreviations are printed below the heatmap: `B=Blur`, `R=Resize`, `N=Noise`, `P=PGD`, `AP=AutoPGD`, `SQ=SquareAttack`; `W=Weak`, `M=Medium`, `E=Extreme`.

The most important comparison for the proposed method is:

```text
Mamba_CLR_0.005  vs.  Mamba_CLR_PCP_Joint_0.005
```

If the method works, `Mamba_CLR_PCP_Joint_0.005` should preserve roughly the same clean validation threshold while improving the post-attack accuracies, especially on `PGD`, `AutoPGD`, and `SquareAttack`.

## Files

- `main.py`: full experiment entry point for the server. By default it uses GPU `0,1,2,3`, trains baselines, trains `Mamba_CLR`, trains the joint `Mamba_CLR_PCP_Joint` method, and runs six robustness tests.
- `config.json`: all server experiment parameters. Edit this file rather than passing long command-line arguments.
- `test.py`: server-side GPU chain test. It reads the `server_test` section in `config.json`, uses a small real MNIST subset, and exercises baseline training, Cross-Lip training, joint CLR+PCP training, all six attack names, JSON export, CSV export, and heatmap creation.
- `plot_only.py`: regenerates a heatmap from an existing `*_summary.json` without training models or rerunning attacks.
- `src/regularizers.py`: Cross-Lipschitz regularizer, distance-aware Plateau-Cliff local norm regularizer, and their joint CLR+PCP regularizer.
- `src/evaluate.py`: blur, resize, noise, PGD, AutoPGD-like multi-restart PGD, and SquareAttack-like black-box evaluation.
- `log/`: all run logs.
- `results/`: JSON, CSV, and heatmap outputs.
- `models/`: checkpoints.
- `artifacts/`: optional saved attacked tensors.

## Server Chain Test

```bash
cd /path/to/thesis_codex/experiment/plateau_cliff
python test.py
```

The test no longer has a CPU or FakeData fallback. It requires CUDA and uses the `server_test` block in `config.json`. By default it runs only a tiny MNIST subset on GPU `0`, so it is meant to check that the server environment, model code, regularizers, attacks, logging, JSON export, and heatmap generation are wired correctly before launching `main.py`.

## Server Run

```bash
cd /path/to/thesis_codex/experiment/plateau_cliff
python main.py
```

To use another config file:

```bash
python main.py --config /path/to/config.json
```

## Plot Only

Use this when the benchmark JSON already exists and you only want to redraw the heatmap:

```bash
cd /disk/user/lwq/workspace/plateau_cliff
python plot_only.py --summary results/plateau_cliff_server_YYYYMMDD_HHMMSS_summary.json
```

By default, this writes next to the summary file with `_heatmap.png` as the suffix. To choose the output path explicitly:

```bash
python plot_only.py \
  --summary results/plateau_cliff_server_YYYYMMDD_HHMMSS_summary.json \
  --output results/redrawn_heatmap.png
```

`--attack-backend art` uses `adversarial-robustness-toolbox` for canonical PGD, AutoPGD, and SquareAttack when it is installed. `--attack-backend torch` uses the built-in PyTorch implementations, which are dependency-light and useful for debugging.

The default `config.json` sets `target_acc` to `0.97` and `stop_at_target` to `true`. Each baseline checkpoint name includes the target, for example `tgt0p97`, and training saves the first checkpoint that reaches the shared clean validation accuracy threshold. This is meant to keep the robustness comparison fair: the models enter attack testing from the same clean-accuracy standard instead of from their individually best clean accuracies.

If `eval_only` is `true`, `main.py` does not train or fine-tune any model. It only loads the expected checkpoints from `models/` and reruns the robustness benchmark with the current `attack_configs`. If a checkpoint is missing, the run fails immediately instead of silently retraining.

The main run writes:

- `results/*_summary.json`
- `results/*_summary.csv`
- `results/*_heatmap.png`
- `results/*_config.json`
- `log/*.log`

## Config Reference

All routine experiment changes should be made in `config.json`. You usually do not need to edit `src/config.py` unless you add a brand-new field that the code does not yet know how to read.

### Run Control

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `profile` | Runtime profile. Use `"server"` for real experiments. | Keep `"server"`. The old CPU smoke profile is no longer used for your workflow. |
| `data_dir` | Root directory passed to `torchvision.datasets.MNIST`. It should contain `MNIST/raw` or `MNIST/processed`. | On your server, use `"/disk/user/lwq/datasets/mnist"`. |
| `download_data` | Whether torchvision may download MNIST if missing. | Keep `false` on the server if data already exists. Set `true` only if the server can access the internet and you want auto-download. |
| `gpus` | CUDA devices visible to this run. | Use `"0,1,2,3"` for the full 4-GPU run. Use `"0"` for quick debugging. |
| `models` | Clean-trained baseline architectures included in the comparison. | Default `["Mamba", "DNN", "nmODE"]`. Removing a model shortens runtime but removes that row from the heatmap. |
| `force_retrain` | Ignore existing checkpoints and train from scratch. | Use `true` only when changing training logic or wanting fresh checkpoints. Keep `false` for normal reruns. |
| `eval_only` | Only load checkpoints and rerun robustness evaluation. No training is allowed. | Use `true` when only changing attacks or plotting style. Set `false` when training new models. |
| `seed` | Random seed for training/evaluation reproducibility. | Change only when you intentionally want another random trial. |

### Training Settings

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `batch_size` | Training batch size. | `128` is reasonable on 4090 GPUs. Increase if memory allows; decrease if OOM occurs. |
| `eval_batch_size` | Batch size for validation and attack evaluation. | `256` is fine for evaluation. Reduce if ART attacks cause memory issues. |
| `num_workers` | DataLoader worker count. `null` lets the profile choose a default. | Keep `null` for full runs. Use `0` if DataLoader multiprocessing causes server issues. |
| `lr` | Adam learning rate. | `0.001` is the default. Lower to `0.0003` if training is unstable. |
| `weight_decay` | Adam weight decay. | Keep `0.0` unless you intentionally add weight decay as another regularizer. |
| `max_epochs` | Safety cap on training epochs. | `30` is enough for MNIST. Increase only if a model cannot reach `target_acc`. |
| `min_epochs` | Minimum epochs before stopping at the target accuracy. | `1` is fine because fairness is controlled by `target_acc`. |
| `target_acc` | Shared clean validation accuracy threshold for entering robustness testing. | `0.97` is a fair first threshold. Try `0.98` or `0.99` for stricter follow-up experiments. |
| `stop_at_target` | Stop when validation accuracy first reaches `target_acc`. | Keep `true` for fairness across models. |

### Dataset Scale

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `train_size` | Number of MNIST training samples used. `null` means full training set. | Keep `null` for final experiments. Use a small integer only for debugging. |
| `valid_size` | Number of test samples used for clean validation. `null` means full test set. | Keep `null` for stable clean-accuracy gating. |
| `phys_eval_size` | Number of samples for non-adversarial attacks: Blur, Resize, Noise. `null` means full test set. | Keep `null` or `10000`; these attacks are relatively cheap. |
| `adv_eval_size` | Number of samples for adversarial attacks. | `3000` is a practical compromise. Increase for stronger evidence, decrease for faster iteration. |

### Method Hyperparameters

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `lambda_clr` | Weight of Cross-Lipschitz regularization. | Default `0.005`. Try nearby values such as `0.003`, `0.008`, `0.01` if doing ablations. |
| `plateau_lambda` | Strength of the Plateau-Cliff local norm shaping term. | Default `0.0005`. Keep conservative; too large may hurt robustness or clean accuracy. |
| `plateau_beta` | Controls how fast the Plateau-Cliff weight decays as distance from the true one-hot label grows. | `8.0` is a reasonable start. Larger means the penalty focuses more sharply on confident points. |
| `cliff_reward` | Optional negative reward near ambiguous/boundary-like points. | Keep `0.0` for safety. Positive values are experimental and may destabilize training. |
| `cliff_tau` | Distance threshold where the cliff gate begins to activate. | Keep `0.35` unless tuning the reward behavior. |
| `cliff_temperature` | Smoothness of the cliff gate around `cliff_tau`. | Keep `0.05`. Smaller is sharper; larger is smoother. |
| `score_space` | Score representation used to compute distance to one-hot labels. | Use `"probability"` by default. `"unit_logits"` is experimental. |

### Attack Backend

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `attack_backend` | Attack implementation for adversarial tests. `"art"` uses Adversarial Robustness Toolbox; `"torch"` uses built-in lightweight attacks. | Use `"art"` for final results. Use `"torch"` for quick debugging or if ART breaks. |
| `save_attack_tensors` | Whether to save attacked images, labels, and predictions. | Keep `false` to save disk. Set `true` only when you need example images or debugging artifacts. |

### Attack Configs

`attack_configs` controls the columns of the heatmap. Each attack has strengths `Weak`, `Medium`, and `Extreme`.

For physical corruptions:

| Attack | Parameter meaning | Current setting advice |
| --- | --- | --- |
| `Blur` | Gaussian blur kernel size. Larger means blurrier. | Current `4/6/8` is moderate. If digits remain too clear, raise to `5/7/9`. |
| `Resize` | Downsample size before resizing back to 28x28. Smaller means stronger information loss. | Current `18/12/8` is moderate. Do not set Extreme too low unless you want nearly unreadable digits. |
| `Noise` | Standard deviation of Gaussian pixel noise. Larger means noisier. | Current `0.07/0.14/0.25` is moderately strong. Values above `0.3` can become visually severe. |

For adversarial attacks:

| Attack | Parameter meaning | Current setting advice |
| --- | --- | --- |
| `PGD.eps` | L-infinity perturbation budget. Larger means stronger attack. | Current `0.04/0.065/0.1` should show differences without collapsing everything. |
| `PGD.steps` | PGD iterations. More steps make the attack stronger and slower. | Current `8/15/30` is a balanced sweep. |
| `AutoPGD.eps` | AutoPGD perturbation budget. | Matched to PGD: `0.04/0.065/0.1`. |
| `AutoPGD.steps` | AutoPGD iterations. | Current `15/30/50`; reduce if runtime is too high. |
| `AutoPGD.n_restarts` | Number of random restarts. More restarts are stronger and slower. | Current `1/2/3`; good for a final-ish run without being excessive. |
| `SquareAttack.eps` | Black-box perturbation budget. | Current `0.07/0.11/0.16`. |
| `SquareAttack.max_iter` | Query/iteration budget. More is stronger and slower. | Current `80/150/250`. |

Good attack settings should not make every model fall to chance-level accuracy in most columns. For MNIST, if many cells are below roughly `20%`, the attack is probably too strong for comparing methods. If nearly all cells are above `95%`, the attack is too weak.

### Server Test Block

The `server_test` section is used only by `test.py`. It overrides the main settings with a tiny GPU run.

| Parameter | Meaning | Setting advice |
| --- | --- | --- |
| `server_test.profile` | Always `"server"` for the server test. | Keep as is. |
| `server_test.gpus` | GPU used for the chain test. | `"0"` is enough. |
| `server_test.models` | Models included in the quick test. | Keep `["Mamba"]` to make the test fast. |
| `server_test.force_retrain` | Retrain tiny test checkpoints every time. | Keep `true`; the test is small and should verify training still works. |
| `server_test.eval_only` | Only evaluate existing tiny checkpoints. | Keep `false` for a real chain test. |
| `server_test.train_size`, `valid_size`, `phys_eval_size`, `adv_eval_size` | Tiny dataset sizes for the server chain test. | Current values are intentionally small. Increase only if debugging statistical behavior. |
| `server_test.attack_backend` | Backend used by the quick test. | Keep `"torch"` so the chain test does not depend on ART. |
| `server_test.attack_configs` | Smoke-strength attacks used by `test.py`. | Keep light; the goal is pipeline verification, not final robustness evidence. |

## Method Summary

Cross-Lip minimizes pairwise class-gradient differences at training points. The new method trains Cross-Lip and the distance-aware input-gradient term jointly:

```text
d(x,y) = 1/2 ||s_theta(x) - e_y||_2^2
w(d) = lambda_p exp(-beta d) - lambda_c sigmoid((d - tau) / T)
L_joint = CE(f_theta(x), y)
        + lambda_clr * CrossLip(f_theta, x)
        + mean_i w(d_i.detach()) ||grad_x s_theta,y_i(x_i)||_2^2
```

When the prediction is close to the true one-hot label, `d` is small and the gradient penalty is strong, encouraging flat plateaus. Near ambiguous points, the penalty decays and can become a small reward if `cliff_reward` is positive, allowing sharper transitions. Cross-Lip stays active during the same training process so that the local norm shaping does not overwrite the robustness gained from class-difference gradient control.
