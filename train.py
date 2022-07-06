from email import parser
from matplotlib import use
from pytorch_lightning.callbacks import QuantizationAwareTraining
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import argparse
from dataset import SSLSegmentationDataset
from trainer import NCPSTrainer, SupervisedTrainer
from torchvision.models.segmentation import deeplabv3_resnet34
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segformer")
    parser.add_argument("--mode", type=str, default="semi")
    parser.add_argument("--model_config", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--use_cutmix", type=bool, default=False)
    parser.add_argument("--pseudo_label_confidence_threshold", type=float, default=0.7)
    parser.add_argument("--use_multiple_teachers", type=bool, default=True)
    parser.add_argument("--prediction_mode", type=str, default="soft_voting")
    parser.add_argument("--labelled_image_dir", type=str, default="datasets/SemiDataset25/labelled/image")
    parser.add_argument("--unlabelled_image_dir", type=str, default="datasets/SemiDataset25/unlabelled/image")
    parser.add_argument("--mask_dir", type=str, default="datasets/SemiDataset25/labelled/mask")
    parser.add_argument("--test_image_dir", type=str, default="datasets/TestDataset/CVC-300/images")
    parser.add_argument("--test_mask_dir", type=str, default="datasets/TestDataset/CVC-300/masks")
    parser.add_argument("--out_dir", type=str, default="datasets/TestDataset/CVC-300/output/semi")
    args = parser.parse_args()

    config = SegformerConfig.from_pretrained(args.model_config)
    config.num_labels = 2
    _test_set = SSLSegmentationDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
        return_image_name=True
    )
    test_set = SSLSegmentationDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    if args.mode == "semi":
        unlabelled_dataset = SSLSegmentationDataset(
            image_dir=args.unlabelled_image_dir,
            input_transform=TRAIN_INPUT_TRANSFORMS,
        )
        labelled_dataset = SSLSegmentationDataset(
            image_dir=args.labelled_image_dir,
            mask_dir=args.mask_dir,
            input_transform=TRAIN_INPUT_TRANSFORMS,
            target_transform=TRAIN_TARGET_TRANSFORMS,
            shared_transform=TRAIN_SHARED_TRANSFORMS,
        )
        print("==== Start training, semi supervised mode ====")
        semi_sup_trainer = NCPSTrainer(
            model_config=config,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            n_labelled_examples_per_batch=BATCH_SIZE // 2,
            use_multiple_teachers=args.use_multiple_teachers,
            use_cutmix=args.use_cutmix,
            pseudo_label_confidence_threshold=args.pseudo_label_confidence_threshold,
        )
        semi_sup_trainer.fit(
            labelled_dataset,
            unlabelled_dataset,
            test_set
        )
        print("Evaluate on test set:")
        print(semi_sup_trainer.evaluate(test_set))
        semi_sup_trainer.predict(_test_set, args.out_dir, args.prediction_mode)
        semi_sup_trainer.save(args.out_dir)

    if args.mode == "small":
        small_labelled_dataset = SSLSegmentationDataset(
            image_dir=args.labelled_image_dir,
            mask_dir=args.mask_dir,
            input_transform=TRAIN_INPUT_TRANSFORMS,
            target_transform=TRAIN_TARGET_TRANSFORMS,
            shared_transform=TRAIN_SHARED_TRANSFORMS,
        )
        print("==== Start training, small supervised mode ====")
        small_sup_trainer = SupervisedTrainer(
            model_config=config,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE
        )
        small_sup_trainer.fit(
            small_labelled_dataset,
        )
        print("Evaluate on test set:")
        print(small_sup_trainer.evaluate(test_set))
        small_sup_trainer.predict(_test_set, args.out_dir)
        small_sup_trainer.save(args.out_dir)
    if args.mode == "full":
        full_labelled_dataset = SSLSegmentationDataset(
            image_dir=args.labelled_image_dir,
            mask_dir=args.mask_dir,
            input_transform=TRAIN_INPUT_TRANSFORMS,
            target_transform=TRAIN_TARGET_TRANSFORMS,
            shared_transform=TRAIN_SHARED_TRANSFORMS,
        )
        print("==== Start training, full supervised mode ====")
        full_sup_trainer = SupervisedTrainer(
            model_config=config,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE
        )
        full_sup_trainer.fit(
            full_labelled_dataset,
        )
        print("Evaluate on test set:")
        print(full_sup_trainer.evaluate(test_set))
        full_sup_trainer.predict(_test_set, args.out_dir)
        full_sup_trainer.save(args.out_dir)
