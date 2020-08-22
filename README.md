** **This is a work in progress** **

<img src="./logo-biobertpr1.png" alt="Logo BioBERTpt">

# BioBERTpt - Portuguese  Clinical BERT

This repository contains fine-tuned [BERT](https://github.com/google-research/bert) models trained on the clinical domain for Portuguese language. BERT-multilingual-cased were trained with clinical narratives from Brazilian hospitals and abstracts of scientific papers from Pubmed and Scielo.

## Download

| Model | PyTorch checkpoint |
|-|:-------------------------:|
|`BioBertPT(all)`  | [Download](https://drive.google.com/open?id=1PrGzj7B0B6rXjPmKoFFOXa1gGjVVHuwA) |
|`BioBertPT(clin)`  | [Download](https://drive.google.com/open?id=1GIOqxPMxeW8sc4EyQ8s1ol3RFWgsBFte) |
|`BioBertPT(bio)`  | [Download](https://drive.google.com/open?id=16D0WA1QMoycvA0tR3KyVdMU1-vpw98sp) |

## Prerequisite
-----
Please download [the huggingface implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

## Usage
-----
1. Download the BioBERTpt (we recomend **BioBERTpt(all)**, see above) and unzip it into <bert_directory>.

2. Install the environment necessary to HuggingFace. 

3. For a NER task, for example, you just need to load the model.

```
model = BertForTokenClassification.from_pretrained(<bert_directory>)
```

4. For more information, you can refer to these [examples](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples).
