import argparse
from dataset import SSLSegmentationDataset
from trainer import NCPSTrainer
from utils import *


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Evaluate n-CPS")
    parser.add_argument("--model_config", type=str, help="Model config. For example: segformer_b0")
    parser.add_argument("--checkpoint_path", type=str, help="The trained model dir")
    parser.add_argument("--prediction_mode", type=str, default="soft_voting", help="Ensembling mode. Set to 'soft_voting', 'max_confidence' or 'single'")
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save the log")
    args = parser.parse_args()

    cvc_300 = SSLSegmentationDataset(
        image_dir="../datasets/TestDataset/CVC-300/images",
        mask_dir="../datasets/TestDataset/CVC-300/masks",
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    cvc_clinic = SSLSegmentationDataset(
        image_dir="../datasets/TestDataset/CVC-ClinicDB/images",
        mask_dir="../datasets/TestDataset/CVC-ClinicDB/masks",
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    colon_db = SSLSegmentationDataset(
        image_dir="../datasets/TestDataset/CVC-ColonDB/images",
        mask_dir="../datasets/TestDataset/CVC-ColonDB/masks",
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    kvasir = SSLSegmentationDataset(
        image_dir="../datasets/TestDataset/Kvasir/images",
        mask_dir="../datasets/TestDataset/Kvasir/masks",
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )
    etis = SSLSegmentationDataset(
        image_dir="../datasets/TestDataset/ETIS-LaribPolypDB/images",
        mask_dir="../datasets/TestDataset/ETIS-LaribPolypDB/masks",
        feature_extractor_config=args.model_config,
        input_transform=VAL_INPUT_TRANSFORMS,
        shared_transform=VAL_SHARED_TRANSFORMS,
    )

    semi_sup_trainer = NCPSTrainer(
        model_config=args.model_config,
        checkpoint_path=args.checkpoint_path,
    )
    print("Evaluate on cvc-300:")
    print(semi_sup_trainer.evaluate(cvc_300, logging_dir=args.out_dir, dataset_alias="cvc-300", mode=args.prediction_mode))
    print("Evaluate on cvc-clinic:")
    print(semi_sup_trainer.evaluate(cvc_clinic, logging_dir=args.out_dir, dataset_alias="cvc-clinic", mode=args.prediction_mode))
    print("Evaluate on cvc-colon:")
    print(semi_sup_trainer.evaluate(colon_db, logging_dir=args.out_dir, dataset_alias="cvc-colon", mode=args.prediction_mode))
    print("Evaluate on etis:")
    print(semi_sup_trainer.evaluate(etis, logging_dir=args.out_dir, dataset_alias="etis", mode=args.prediction_mode))
    print("Evaluate on kvasir:")
    print(semi_sup_trainer.evaluate(kvasir, logging_dir=args.out_dir, dataset_alias="kvasir", mode=args.prediction_mode))