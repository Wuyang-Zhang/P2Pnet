import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate basic regression metrics for point detection/counting tasks"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE) - avoid division by zero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0

    return mse, mae, rmse, r2, mape


def calculate_average_precision(y_true, y_pred):
    """Calculate mean absolute percentage error (custom implementation)"""
    total_precision = 0
    count = 0
    for real, pred in zip(y_true, y_pred):
        if real != 0:  # 避免除以零
            precision = abs(pred - real) / real
            total_precision += precision
            count += 1
    # 计算平均精度
    average_precision = total_precision / count if count else 0
    return average_precision

def calculate_detection_metrics(y_true, y_pred, threshold=0.1):
    """
    Calculate detection-specific metrics for point detection tasks

    Args:
        y_true: Ground truth counts
        y_pred: Predicted counts
        threshold: Relative error threshold for "correct detection" (default 10%)

    Returns:
        dict: Dictionary containing various detection metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy within threshold (percentage of predictions within threshold of true value)
    relative_errors = np.abs(y_pred - y_true) / (y_true + 1e-8)  # Add small epsilon to avoid division by zero
    accuracy_within_threshold = np.mean(relative_errors <= threshold) * 100

    # Count accuracy (exact matches)
    exact_matches = np.sum(y_pred == y_true)
    count_accuracy = exact_matches / len(y_true) * 100

    # Underestimation/Overestimation metrics
    underestimation = np.mean(y_pred < y_true)
    overestimation = np.mean(y_pred > y_true)

    # Mean relative error
    mean_relative_error = np.mean(relative_errors) * 100

    # Standard deviation of errors
    errors = y_pred - y_true
    error_std = np.std(errors)

    return {
        'accuracy_within_threshold': accuracy_within_threshold,
        'count_accuracy': count_accuracy,
        'underestimation_rate': underestimation * 100,
        'overestimation_rate': overestimation * 100,
        'mean_relative_error': mean_relative_error,
        'error_std': error_std
    }

def calculate_counting_ranges(y_true, y_pred, ranges=[(0, 10), (10, 50), (50, 100), (100, float('inf'))]):
    """
    Calculate metrics for different counting ranges
    Useful for analyzing performance across different crowd density levels

    Args:
        y_true: Ground truth counts
        y_pred: Predicted counts
        ranges: List of tuples defining count ranges

    Returns:
        dict: Metrics for each range
    """
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for range_min, range_max in ranges:
        mask = (y_true >= range_min) & (y_true < range_max)
        if np.any(mask):
            range_true = y_true[mask]
            range_pred = y_pred[mask]

            mse = mean_squared_error(range_true, range_pred)
            mae = mean_absolute_error(range_true, range_pred)

            range_name = f"{range_min}-{range_max if range_max != float('inf') else 'inf'}"
            results[range_name] = {
                'count': len(range_true),
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse)
            }

    return results

def read_data_from_file(file_path):
    y_pred = []
    y_true = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                predicted = float(parts[1])
                actual = float(parts[2])
                y_pred.append(predicted)
                y_true.append(actual)

    return y_true, y_pred



def print_comprehensive_metrics(y_true, y_pred):
    """Print comprehensive evaluation metrics for point detection/counting tasks"""
    print("=" * 60)
    print("COMPREHENSIVE POINT DETECTION EVALUATION METRICS")
    print("=" * 60)

    # Basic regression metrics
    mse, mae, rmse, r2, mape = calculate_metrics(y_true, y_pred)
    print("\n📊 BASIC REGRESSION METRICS:")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    # Detection-specific metrics
    detection_metrics = calculate_detection_metrics(y_true, y_pred)
    print("\n🎯 DETECTION-SPECIFIC METRICS:")
    print(f"Accuracy within 10% threshold: {detection_metrics['accuracy_within_threshold']:.2f}%")
    print(f"Exact count accuracy: {detection_metrics['count_accuracy']:.2f}%")
    print(f"Underestimation rate: {detection_metrics['underestimation_rate']:.2f}%")
    print(f"Overestimation rate: {detection_metrics['overestimation_rate']:.2f}%")
    print(f"Mean relative error: {detection_metrics['mean_relative_error']:.2f}%")
    print(f"Error standard deviation: {detection_metrics['error_std']:.4f}")
    # Counting range analysis
    range_metrics = calculate_counting_ranges(y_true, y_pred)
    print("\n📈 PERFORMANCE BY COUNTING RANGES:")
    for range_name, metrics in range_metrics.items():
        print(f"  Range {range_name} (n={metrics['count']}):")
        print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

    # Legacy metric for compatibility
    cprecision = calculate_average_precision(y_true, y_pred)
    print(f"\n🔄 LEGACY METRIC:")
    print(f"Custom Precision (cprecision): {cprecision:.4f}")
    print("=" * 60)

if __name__ == "__main__":

    file_path = './output/pre_gd_cnt.txt'
    try:
        y_true, y_pred = read_data_from_file(file_path)
        if not y_true:
            print(f"Error: No data found in {file_path}")
            exit(1)

        print(f"Loaded {len(y_true)} samples from {file_path}")
        print_comprehensive_metrics(y_true, y_pred)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        exit(1)