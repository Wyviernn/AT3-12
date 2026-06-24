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

    # Predict Alex
    student_name = 'Alex Anderson'
    try:
        pred = p.predict_student(df, student_name)
        print(f"Predicted final score for {student_name}: {pred:.2f}")
        # Save result
        out_path = os.path.join(os.path.dirname(__file__), 'prediction_result.txt')
        with open(out_path, 'w') as f:
            f.write(f"Student: {student_name}\nPredicted {p.target_column}: {pred:.2f}\nRMSE: {metrics['rmse']:.3f}\n")
        print(f"Prediction written to {out_path}")
    except Exception as e:
        print('Prediction failed:', e)


if __name__ == '__main__':
    main()
