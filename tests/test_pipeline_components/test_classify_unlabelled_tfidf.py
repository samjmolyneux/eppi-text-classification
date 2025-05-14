import datetime
from pathlib import Path

from pipelines.final_components import PredArgs, run_tfidf_classification

if __name__ == "__main__":
    dirs = [
        d
        for d in Path("./sim_pipeline_outputs/find_single_model/").iterdir()
        if d.is_dir()
    ]
    # Most recent outputs from single model is input dir
    input_dir = max(dirs, key=lambda d: d.stat().st_mtime)

    current_time = datetime.datetime.now(datetime.UTC).strftime("%S-%M-%H-%d-%m-%Y")
    pred_labels_dir = (
        f"./sim_pipeline_outputs/classify_unlabelled_tfidf/{current_time}/pred_labels"
    )

    args = PredArgs(
        unlabelled_data_dir=str(input_dir / "unlabelled_tfidf"),
        threshold=3.4,
        model_dir=str(input_dir / "trained_model"),
        pred_labels_dir=pred_labels_dir,
    )
    Path.mkdir(Path(pred_labels_dir), parents=True, exist_ok=True)
    run_tfidf_classification(args)
