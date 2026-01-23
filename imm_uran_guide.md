# Установка окружения
Для установки окружения достаточно запустить скрипть `install.sh` из корня проекта.

После успешного выполнения скрипта будут созданы две виртуальные среды Conda:
1. Виртуальная среда для обучения классификатора vit-p (активируется командой `conda activate vitp`)
2. Виртуальная среда для обучения генератора сегментационных масок oneformer (`conda activate --prefix ./OneFormer/env`)

## Обучение ViT-P на наборе ADE20k
Обучение можно запустить с помощью скрипта `train_ADE20k.sh`
`ADEChallengeData2016.zip` можно найти в папке `/home6/m_imm_freedata/Segmentation/Projects`.  

## Запуск OneFormer + ViT-P на наборе ADE20k
Переходим в директорию `OneFormer`. Для запуска модели достаточно запустить скрипт `evaluate.sh`.

Готовый набор с разметкой для семантической сегментации лежит в папке `/home6/m_imm_freedata/Segmentation/Projects/lapshin/Detectron2_datasets`. Разметка была сгенерирована с помощью скрипта `./OneFormer/datasets/prepare_ade20k_sem_seg.py`.