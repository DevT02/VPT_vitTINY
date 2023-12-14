# Visual Prompt tuning on TinyViT with Backdoor

Why are Backdoors with VPT on TinyViT a good thing?

Actually, we can significantly improve a model's performance. We can use FGSM with loss to calculate how we should account for scenarios that are altered in ways that we would have otherwise known. We can accomplish such a task through visual prompt tuning.

Why are they a bad thing?

It opens a loophole for security risks. If we don't add the perturbations in a manner that people cannot exploit, malicious actors could use this to trigger misclassification.

![An example of why this is bad](https://github.com/DevT02/VPT_vitTINY/assets/40608267/c4cf9a79-ff9b-4915-95ba-4835c57a1b6f)

## Setup

Using Microsoft's TinyViT implementation. It's already in the repo.

<details open>
<summary> Microsoft's Official TinyViT implementation. </summary>

```git
git clone https://github.com/microsoft/Cream
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install chardet
```
Ensure conda is installed for best performance. Chardet may be required.
</details>

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic training:
```bash
python main_vpt_tinyvit.py --dataset CIFAR10 --prompt_tokens 5
```

With adversarial training:
```bash
python main_vpt_tinyvit.py --dataset CIFAR10 --use_adv_training --epsilon 0.1
```

With backdoor:
```bash
python main_vpt_tinyvit.py --dataset CIFAR10 --use_backdoor --backdoor_target 0
```

All together:
```bash
python main_vpt_tinyvit.py --dataset CIFAR10 --prompt_tokens 5 --use_adv_training --epsilon 0.1 --use_backdoor --backdoor_target 0
```

## Arguments

`--dataset`: CIFAR10 or MNIST

`--batch_size`: batch size (default 32)

`--epochs`: number of epochs (default 10, early stopping at 5)

`--lr`: learning rate (default 1e-3)

`--prompt_tokens`: number of prompt tokens (default 5)

`--prompt_type`: shallow or deep (default shallow)

`--prompt_dropout`: dropout for prompts (default 0.0)

`--use_adv_training`: enable FGSM adversarial training

`--epsilon`: FGSM perturbation magnitude (default 0.1)

`--adv_prob`: probability of using adversarial examples (default 0.5)

`--use_backdoor`: enable backdoor trigger

`--backdoor_target`: target class for backdoor (default 0)

`--backdoor_prob`: probability of backdoor injection (default 0.3)

`--img_size`: image size (default 224)

`--seed`: random seed (default 42)

`--data_dir`: data directory (default ./data)

`--save_dir`: checkpoint directory (default ./checkpoints)

`--results_dir`: results directory (default ./results)

`--no_track_results`: disable automatic results tracking (enabled by default)

`--early_stop_patience`: epochs without improvement before stopping (default 5)

## Files

`main_vpt_tinyvit.py`: main training script

`vpt_tinyvit.py`: VPT implementation

`vpt_tinyvit_simple.py`: simpler VPT version

`attacks.py`: FGSM and backdoor code

`results_tracker.py`: automatic experiment tracking

`Cream_git/TinyViT/`: TinyViT implementation

## Outputs

Saves to `checkpoints/`:

`best_model.pth`: best model checkpoint

`training_curves.png`: training plots (if not using results tracker)

Automatically saves to `results/` (unless `--no_track_results` is used):

Each experiment gets its own timestamped folder

`config.json`: experiment configuration

`summary.json`: results summary with best metrics

`training_curves.png`: training plots

`all_results.json`: master file with all experiments

Summary is shown automatically at the end of training. Check `results/all_results.json` to compare all experiments.

## Results

Tested on CIFAR10 with VPT + FGSM + backdoor:

Best validation accuracy: 85.13% at epoch 7

Test accuracy: 79.65%

Backdoor accuracy: 47.07%

The model peaked at epoch 7 then started degrading. Early stopping would have saved time. Validation accuracy was consistently higher than training accuracy, which is normal for VPT since the backbone is frozen and pretrained; the prompts are still learning while the backbone already knows what to do.

Backdoor worked partially. 47% is way above random (10%) but not perfect. Could probably improve with more backdoor training or different trigger patterns.

For VPT on CIFAR10, 79% test accuracy is pretty solid. Could probably push it to 82-85% with better hyperparameters or more prompt tokens.

## Interpreting Results

Best Val Acc is the highest validation accuracy during training; that's what gets saved in the checkpoint. Test Acc is the real performance metric on unseen data. Backdoor Acc shows if the backdoor trigger actually works.

What to look for:

Best Val Acc > Test Acc might mean overfitting; Test Acc ~ Best Val Acc means good generalization

Val Acc > Train Acc is normal for VPT; backbone is frozen/pretrained, prompts learning

Backdoor Acc much higher than random (10% for CIFAR10) means backdoor is working

Early stopping triggered means model stopped improving; saved time

## Notes

Backbone is frozen during VPT training; only prompts and head are trainable

Adversarial training improves robustness but may reduce clean accuracy

Backdoor triggers are for research purposes only

Tested on RTX 3060; batch size and epochs were limited by GPU memory

Early stopping is enabled by default (stops after 5 epochs without improvement)

## For VPT test

<details open>
<summary> VIT with VPT (including MSFT's) </summary>

```git
pip install iopath simplejson ml-collections fvcore pandas
```
</details>

References: 
```BibTeX
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}

@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
