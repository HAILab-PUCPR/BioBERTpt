** **This is a work in progress** **

<img src="./logo-biobertpr1.png" alt="Logo BioBERTpt">

# BioBERTpt - Portuguese Clinical and Biomedical BERT

This repository contains fine-tuned [BERT](https://github.com/google-research/bert) models trained on the clinical domain for Portuguese language. Pre-trained BERT-multilingual-cased were fine-tuned with clinical narratives from Brazilian hospitals and abstracts of scientific papers from Pubmed and Scielo.

## NER Experiment in SemClinBr Corpora

We evaluate our models on [SemClinBr](https://github.com/HAILab-PUCPR/SemClinBr), a semantically annotated corpus for Portuguese clinical NER, containing 1,000 labeled clinical notes. These corpus comprehended 100 UMLS semantic types, summarized in 13 groups of entities: Disorders, Chemicals and Drugs, Medical Procedure, Diagnostic Procedure, Disease Or Syndrome, Findings, Health Care Activity, Laboratory or Test Result, Medical Device, Pharmacologic Substance, Quantitative Concept, Sign or Symptom and Therapeutic or Preventive Procedure.

The table below shows complete **F1-score** results for each entity in SemClinBr, where the last three models (in italian) are our in-domain models. In bold, the higher values. 

| Entity / Model | Disorders | ChemicalDrugs | Procedures | DiagProced | DiseaseSynd | Findings | Heatlh | Laboratory | Medical | Pharmacologic | Quantitative | Sign | Therapeutic |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|BERT multilingual uncased|0.787|0.903|0.670|0.546|0.562|0.503|0.374|0.378|0.559|0.756|**0.607**|0.519|0.487|
|BERT multilingual cased|0.782|0.901|0.675|0.519|0.538|0.505|0.412|0.417|0.593|0.593|0.613|0.537|0.486|
|Portuguese BERT large|0.625|0.782|0.453|0.504|0.575|**0.526**|0.336|0.404|0.514|0.723|0.562|0.552|0.489|
|Portuguese BERT base|0.784|0.904|0.672|0.556|0.5400|0.500|0.346|0.422|0.537|0.775|0.568|0.538|0.471|
|*BioBERtpt (bio)*|0.785|0.894|0.689|0.550|0.575|0.526|**0.459**|0.398|**0.604**|0.724|0.592|0.534|0.501|
|*BioBERtpt (clin)*|0.781|**0.911**|0.686|**0.560**|**0.583**|0.521|0.406|**0.453**|0.562|**0.779**|0.593|0.544|0.459|
|*BioBERtpt (all)*|**0.791**|0.904|**0.703**|0.548|0.564|0.517|0.403|0.440|0.555|0.747|0.600|**0.566**|**0.513**|

## Download BioBERTpt

| Model | Domain | PyTorch checkpoint | 
|------|-------|:-------------------------:|
|`BioBERTpt (all)`  | Clinical + Biomedical |  [Download](https://drive.google.com/open?id=1PrGzj7B0B6rXjPmKoFFOXa1gGjVVHuwA) |
|`BioBERTpt (clin)`  | Clinical | [Download](https://drive.google.com/open?id=1GIOqxPMxeW8sc4EyQ8s1ol3RFWgsBFte) |
|`BioBERTpt (bio)`  | Biomedical | [Download](https://drive.google.com/open?id=16D0WA1QMoycvA0tR3KyVdMU1-vpw98sp) |

## Prerequisite
-----
Please download the amazing [Huggingface implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

## Usage
-----
1. Download the BioBERTpt (we recomend **BioBERTpt(all)**, see above) and unzip it into <bert_directory>.

2. Install the environment necessary to HuggingFace. 

3. For a NER task, for example, you just need to load the model.

```
model = BertForTokenClassification.from_pretrained(<bert_directory>)
```

For more information, you can refer to these [examples](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples).

## Reproduce BioBERTpt
-----

To replicate our work, or fine-tune you own model, just do this steps:

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

mkdir data

# please put your corpus file in this folder in a txt format

python examples/run_language_modeling.py --output_dir=output --model_type=bert \
    --model_name_or_path=bert-base-multilingual-cased --do_train --train_data_file=data/corpus.txt  --num_train_epochs 15 --mlm \
	--learning_rate 1e-5  --per_gpu_train_batch_size 16 --seed 666 --block_size=512
```

## Citation

**(soon)**
