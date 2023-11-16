# Visual Prompt tuning on TinyViT with Backdoor

Why are Backdoors with VPT on TinyViT a good thing?

Actually, we can significantly improve a model's performance. We can use FGSM with loss to calculate how we should account for scenarios that are altered in ways that we would have otherwise known. We can accomplish such a task through visual prompt tuning.

Why are they a bad thing?

It opens a loophole for security risks. If we don't add the perturbations in a manner that people cannot exploit, malicious actors could use this to trigger misclassification.

![An example of why this is bad](https://github.com/DevT02/VPT_vitTINY/assets/40608267/c4cf9a79-ff9b-4915-95ba-4835c57a1b6f)


## At the moment, use MSFT's

<details open>
<summary> Microsoft's Official TinyViT implementation. </summary>

```git
git clone https://github.com/microsoft/Cream
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install chardet
```
Ensure conda is installed for best performance. Chardet may be required.
</details>

## Try this model (for later)
<details open>
<summary> Google's VIT </summary>


```git
git clone https://github.com/google-research/vision_transformer
```
</details>

## For VPT test
<details open>
<summary> VIT with VPT (including MSFT's) </summary>

```git
pip install iopath simplejson ml-collections
```
</details>


References: 
```BibTeX
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}

@InProceedings{vpl,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2023}
}

```
