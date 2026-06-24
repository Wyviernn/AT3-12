import os
import importlib.util


def load_predictor_module():
    path = os.path.join(os.path.dirname(__file__), 'predictor.py')
    spec = importlib.util.spec_from_file_location('predictor', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    csv_path = os.path.join(repo_root, 'Data', 'data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    predictor_mod = load_predictor_module()
    p = predictor_mod.UniversalPredictor()

    print("Loading and cleaning data...")
    df = p.load_data(csv_path)

    print("Training models for all missing columns...")
    p.train_models(df)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70)
    for col, metrics in p.metrics.items():
        print(f"\n{col}:")
        print(f"  Training samples: {metrics['train_size']}")
        print(f"  Test samples: {metrics['test_size']}")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        print(f"  Cross-validation R² Score: {metrics['cv_r2_score']:.4f}")
    print("\n" + "=" * 70)

    print("\nFinding students with missing values...")
    missing_students = p.get_all_missing_students(df)

    if not missing_students:
        print("No students with missing values found.")
        return

    print(f"Found {len(missing_students)} student(s) with missing data.\n")
    print("=" * 60)

    results = []
    for student_name in missing_students:
        predictions = p.predict_missing_values(df, student_name)
        if predictions:
            results.append((student_name, predictions))

    # Print in requested S.U.N Academic predictor format
    print('\nS.U.N Academic predictor')
    print('-' * 91)

    output_file = os.path.join(os.path.dirname(__file__), 'PREDICTIONS.txt')
    with open(output_file, 'w') as f:
        f.write('S.U.N Academic predictor\n')
        f.write('-' * 91 + '\n')

        for student_name, predictions in results:
            print(student_name)
            f.write(student_name + '\n')
            for col, pred_value in predictions.items():
                formatted_pred = f"{pred_value:.2f}"
                print(f"{col}: {formatted_pred}")
                f.write(f"{col}: {formatted_pred}\n")
            print('-' * 13)
            f.write('-' * 13 + '\n')

    print('\nPredictions complete.')
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
