import os
import importlib.util


def load_predictor_module():
    path = os.path.join(os.path.dirname(__file__), 'predictor.py')
    spec = importlib.util.spec_from_file_location('predictor_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    csv_path = os.path.join(repo_root, 'Markbook', 'master_markbook - master_markbook.csv')

    predictor = load_predictor_module()
    p = predictor.AcademicPredictor()
    df = p.load_data(csv_path)

    print('Training model...')
    metrics = p.train(df)
    print(f"Training complete. RMSE on hold-out set: {metrics['rmse']:.3f}")
    # Predict all students with any missing score in their row
    df_clean = p.clean_data(df)
    missing_mask = df_clean.drop(columns=['Student_Name']).isna().any(axis=1)
    missing = df_clean[missing_mask]

    if missing.empty:
        print('No students with missing data found.')
        return

    preds = []
    for _, row in missing.iterrows():
        student_name = row['Student_Name']
        X_row = row.drop(labels=[p.target_column, 'Student_Name'], errors='ignore').to_frame().T
        X_row = X_row.reindex(columns=p.feature_columns)
        try:
            pred_val = float(p.pipeline.predict(X_row)[0])
        except Exception:
            pred_val = None
        preds.append({'Student_Name': student_name, 'Predicted_Final': pred_val})

    out_csv = os.path.join(os.path.dirname(__file__), 'predictions_missing_any_nan.csv')
    import csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Student_Name', 'Predicted_Final'])
        writer.writeheader()
        for r in preds:
            writer.writerow(r)

    print(f"Predicted {len(preds)} rows with missing data. Saved to {out_csv}")


if __name__ == '__main__':
    main()
