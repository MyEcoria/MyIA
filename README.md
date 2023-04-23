# MyIA

## Bard Mix:
Il faut placer les models dans le dossier models

`g++ mix_models.cpp -o mix_models`

`./mix_models`

## Nano_node:
`g++ -o train_gpt4all train_gpt4all.cpp`

`./train_gpt4all`

## Train GPT4all in c++

```bash

   sudo apt-get install libtorch-dev
   sudo apt-get install libtorch-cpu-dev

```

`g++ -o train train.cc`
`./train --config configs/train/finetune-7b.yaml`
