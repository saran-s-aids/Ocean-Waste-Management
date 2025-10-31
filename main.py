import argparse
from pathlib import Path

from src import config
from src.train import train
from src.evaluate import evaluate
from src.realtime import realtime_demo


def main():
    parser = argparse.ArgumentParser(description="Marine Plastic Waste Detection & Classification")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to dataset root with class subfolders")
    parser.add_argument("--img_size", type=int, nargs=2, default=list(config.IMG_SIZE), help="Image size H W")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--validation_split", type=float, default=config.VALIDATION_SPLIT)
    parser.add_argument("--seed", type=int, default=config.SEED)

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--realtime", action="store_true", help="Run real-time webcam demo")

    parser.add_argument("--model_path", type=str, default=str(config.MODEL_PATH), help="Path to .h5 model file")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_size = (args.img_size[0], args.img_size[1])

    model = None
    val_ds = None
    class_names = None

    if args.train:
        model, val_ds, class_names, _ = train(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            seed=args.seed,
        )

    if args.evaluate:
        # Use existing val_ds if already created during training
        if val_ds is not None and class_names is not None:
            evaluate(model_path=args.model_path, val_ds=val_ds, class_names=class_names)
        else:
            evaluate(model_path=args.model_path)

    if args.realtime:
        realtime_demo(model_path=args.model_path, img_size=img_size)

    # Default: if no flag is passed, run train then evaluate
    if not (args.train or args.evaluate or args.realtime):
        model, val_ds, class_names, _ = train(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            seed=args.seed,
        )
        evaluate(model_path=args.model_path, val_ds=val_ds, class_names=class_names)


if __name__ == "__main__":
    main()
