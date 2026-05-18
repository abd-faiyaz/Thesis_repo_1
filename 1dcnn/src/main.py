import pandas as pd
from pathlib import Path
from model.train import train_model
from model.evaluate import evaluate_model
from logger import get_logger


logger = get_logger("Pipeline")


def run_baseline():
    """
    Main pipeline.
    Experiment: Train on 2020, Test on 2021, 2022, 2023 (individually)
    """

    data_root = Path("data")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    train_year = "2020"
    test_years = ["2021", "2022", "2023"]

    experiment_name = "bytecnn_basemodel_2020"

    train_data_path = data_root / train_year
    model_path = model_dir / f"{experiment_name}.pth"

    logger.info(f"Training on year: {train_year}")
    logger.info(f"Training data: {train_data_path}")
    logger.info(f"Model will be saved to: {model_path}")

    BYTE_LENGTH = 1024
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 0.001
    FROM_END=True
    THRESHOLD = 0.5

    train_model(
        year_dir=train_data_path,
        model_save_path=model_path,
        byte_length=BYTE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LR,
        from_end=FROM_END,
        verbose=True
    )

    logger.info("Training completed.")

    results = []

    for test_year in test_years:
        test_data_path = data_root / test_year

        logger.info(f"Evaluating on year: {test_year}")
        logger.info(f"Test data: {test_data_path}")

        metrics = evaluate_model(
            year_dir=test_data_path,
            model_path=model_path,
            report_path=f"reports/{test_year}_result.csv",
            byte_length=BYTE_LENGTH,
            threshold=THRESHOLD,
            from_end=FROM_END
        )

        results.append({
            "year": test_year,
            **metrics
        })

        logger.info(f"Evaluation for {test_year} completed.")
    
    df = pd.DataFrame(results)
    df.to_csv("reports/summary.csv", index=False)


def run_sliding_window():
    return


def main():
    logger.info("1D-CNN Training & Evaluation Started.......")
    run_baseline()
    run_sliding_window()
    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()
