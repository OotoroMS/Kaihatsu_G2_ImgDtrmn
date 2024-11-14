import os
import json
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import re
from PIL import Image

# ディレクトリの設定
DAMAGED_IMAGES_DIR = "./TestImages/defective_images"    # 不良品画像のディレクトリ
PASSED_IMAGES_DIR = "./TestImages/flawless_images"      # 良品画像のディレクトリ
RESULTS_DIR = "./results"
OUTPUT_IMAGES_DIR = "./output_images"     # 出力画像を保存するディレクトリ

# ディレクトリが存在しない場合は作成
for directory in [RESULTS_DIR, OUTPUT_IMAGES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model_and_info(model_path, work_name):
    """モデルを読み込み、パラメータ数とエポック数を取得する関数"""
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"\nエラー発生: モデル '{model_path}' の読み込みに失敗しました。詳細: {e}")
        return None, None, None, None, None

    # モデル名から情報を抽出
    base_name = os.path.basename(model_path)
    pattern = r'^(.+?)_(simple|noble|advanced|massive|infinity)_(full|half|quarter)_epoch_(\d{3})\.keras$'
    match = re.search(pattern, base_name)
    if match:
        work_name = match.group(1)
        model_structure = match.group(2)
        io_size = match.group(3)
        epoch_str = match.group(4)
        epochs = int(epoch_str)
    else:
        print(f"モデル名 '{base_name}' が命名規則に合致しません。")
        epochs = None

    parameter_count = model.count_params()

    # モデルのタイプを判定
    if "simple" in base_name:
        model_type = "simple"
    elif "advanced" in base_name:
        model_type = "advanced"
    elif "massive" in base_name:
        model_type = "massive"
    elif "infinity" in base_name:
        model_type = "infinity"
    else:
        model_type = "unknown"

    # ワーク名とモデルタイプを使用してモデル名をフォーマット
    formatted_model_name = f"{work_name}_{model_type}_epoch_{str(epochs).zfill(3)}.keras"

    return model, model_type, epochs, parameter_count, formatted_model_name

def robust_scale(image_array):
    """
    ロバストスケーリングを適用する関数
    :param image_array: スケーリング前の画像配列（numpy.ndarray）
    :return: スケーリング後の画像配列（numpy.ndarray）
    """
    median = np.median(image_array)
    q75, q25 = np.percentile(image_array, [75 ,25])
    iqr = q75 - q25
    if iqr == 0:
        print("警告: IQRが0のため、スケーリングを中央値のみで実施します。")
        return image_array - median
    scaled = (image_array - median) / iqr
    return scaled

def preprocess_image(image_path, crop_box, input_shape):
    """画像を読み込み、前処理（クロップとロバストスケーリング）を行う関数"""
    try:
        image = load_img(image_path, color_mode='grayscale')
        image = image.crop(crop_box)  # クロップ処理
        image = image.resize((input_shape[1], input_shape[0]))  # PILではサイズが(width, height)なので注意
        image_array = img_to_array(image).astype(np.float32)
        # ロバストスケーリングの適用
        image_array = robust_scale(image_array)
        # 正規化（必要に応じて調整）
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
        image_array = np.expand_dims(image_array, axis=0)  # バッチ次元を追加
        return image_array
    except Exception as e:
        print(f"\nエラー発生: {image_path} の読み込みまたは前処理に失敗しました。詳細: {e}")
        return None

def save_output_image(output_image, output_path, input_image_path):
    """出力画像を保存する関数"""
    try:
        # 画像を正規化から元に戻して保存
        output_image = np.squeeze(output_image, axis=0)  # バッチ次元を削除
        output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)  # 0-1の範囲を0-255に戻す
        output_img = array_to_img(output_image)
        output_img.save(output_path)
    except Exception as e:
        print(f"\nエラー発生: 出力画像の保存に失敗しました。詳細: {e}")

def calculate_mae(input_image_path, output_image_path):
    """入力画像と出力画像のMAEを計算する関数"""
    try:
        input_image = Image.open(input_image_path).convert('L')  # グレースケール
        output_image = Image.open(output_image_path).convert('L')  # グレースケール
        # 画像サイズの確認
        if input_image.size != output_image.size:
            # 出力画像を入力画像のサイズにリサイズ
            output_image = output_image.resize(input_image.size)
        input_array = np.array(input_image).astype(np.float32)
        output_array = np.array(output_image).astype(np.float32)
        mae = np.mean(np.abs(input_array - output_array))
        return mae
    except Exception as e:
        print(f"\nエラー発生: MAEの計算に失敗しました。詳細: {e}")
        return None

def inference(model, image, image_name, work_name, epoch, label, formatted_model_name, image_path):
    """推論を行い、出力画像を保存し、MAEを計算する関数"""
    start_time = time.time()
    try:
        output = model.predict(image)
    except Exception as e:
        print(f"\nエラー発生: モデル '{formatted_model_name}' の推論に失敗しました。詳細: {e}")
        return None, None, None
    end_time = time.time()

    # 出力画像を保存するパス
    model_output_dir = os.path.join(OUTPUT_IMAGES_DIR, formatted_model_name.replace('.keras', ''))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # 出力画像の名称を「[使用しているモデル名]_[入力画像名].jpg」に変更
    output_image_name = f"{formatted_model_name.replace('.keras', '')}_{image_name}.jpg"
    output_image_path = os.path.join(model_output_dir, output_image_name)

    save_output_image(output, output_image_path, image_path)
    
    # MAEの計算
    mae = calculate_mae(input_image_path=image_path, output_image_path=output_image_path)
    
    inference_time = end_time - start_time
    
    if mae is None:
        return None, inference_time, output_image_path  # MAE計算に失敗した場合はスキップ
    
    return float(mae), float(inference_time), output_image_path  # MAEと推論時間を返す

def benchmark_images(model, image_dir, label, crop_box, work_name, epoch, input_shape, formatted_model_name):
    """指定されたディレクトリの画像を使用して推論を実行し、結果を収集する関数"""
    results = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            continue  # ファイルでない場合はスキップ
        image = preprocess_image(image_path, crop_box, input_shape)
        if image is None:
            continue  # 前処理に失敗した場合はスキップ
        mae, inference_time, output_image_path = inference(model, image, image_name, work_name, epoch, label, formatted_model_name, image_path)
        
        if mae is None:
            continue  # MAEの計算に失敗した場合はスキップ
        
        # 推論結果を追加
        results.append({
            "image_path": image_path,
            "output_image_path": output_image_path,
            "mae": mae,
            "inference_time": inference_time,
            "label": label  # "good" または "bad" のラベル
        })
    return results

def main():
    # ユーザーからワーク名を入力
    work_name = input("ワーク名を入力してください: ").strip()
    if not work_name:
        print("エラー: ワーク名が入力されていません。")
        return

    # JSONファイルのパスを設定
    config_file = f"{work_name}.json"

    # JSONファイルの存在を確認
    if not os.path.exists(config_file):
        print(f"エラー: 設定ファイル '{config_file}' が見つかりません。")
        return

    # JSONファイルの読み込み
    with open(config_file, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"エラー: 設定ファイル '{config_file}' の読み込みに失敗しました。詳細: {e}")
            return

    # クロップ範囲の取得
    try:
        crop_ranges = config['crop_ranges']
        x_start = crop_ranges['x_start']
        x_end = crop_ranges['x_end']
        y_start = crop_ranges['y_start']
        y_end = crop_ranges['y_end']
        crop_box = (x_start, y_start, x_end, y_end)  # (left, upper, right, lower)
    except KeyError as e:
        print(f"エラー: 設定ファイル '{config_file}' に必要なキー {e} が存在しません。")
        return

    # モデルディレクトリのパスを設定
    model_dir = f"./AE_model_{work_name}"

    # モデルディレクトリの存在を確認
    if not os.path.exists(model_dir):
        print(f"エラー: モデルディレクトリ '{model_dir}' が見つかりません。")
        return

    # 結果を格納するディレクトリを作成
    work_results_dir = os.path.join(RESULTS_DIR, work_name)
    if not os.path.exists(work_results_dir):
        os.makedirs(work_results_dir)

    # モデル構造の選択
    print("\nベンチマークしたいモデル構造を選択してください。")
    print("利用可能な構造: simple, advanced, massive, infinity")
    print("すべての構造をベンチマークする場合は 'all' と入力してください。")
    selected_structures_input = input("-> ").strip().lower()
    if selected_structures_input == "all":
        selected_structures = ["simple", "advanced", "massive", "infinity"]
    else:
        selected_structures = [s.strip() for s in selected_structures_input.split(",")]
        valid_structures = ["simple", "advanced", "massive", "infinity"]
        for s in selected_structures:
            if s not in valid_structures:
                print(f"エラー: 無効なモデル構造 '{s}' が指定されました。")
                return

    # モデルディレクトリからモデルをロード
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras") and any(struct in f for struct in selected_structures)]
    if not model_files:
        print(f"エラー: 指定されたモデル構造の '.keras' ファイルがモデルディレクトリ '{model_dir}' に存在しません。")
        return

    # モデルファイルをエポック順にソート
    def extract_epoch(model_name):
        match = re.search(r'epoch_(\d{3})\.keras$', model_name)
        if match:
            return int(match.group(1))
        else:
            return -1  # エポック数が取得できない場合は-1を返す

    model_files.sort(key=extract_epoch)

    for model_name in model_files:
        model_path = os.path.join(model_dir, model_name)
        loaded = load_model_and_info(model_path, work_name)
        if loaded is None:
            continue  # モデルの読み込みに失敗した場合はスキップ
        model, model_type, epochs, parameter_count, formatted_model_name = loaded
        if epochs is None:
            print(f"警告: モデル '{model_name}' からエポック数を抽出できませんでした。スキップします。")
            continue

        print(f"\nモデル '{formatted_model_name}' をロードしました。タイプ: {model_type}, エポック数: {epochs}, パラメータ数: {parameter_count}")

        # モデルの入力形状を取得
        input_shape = model.input_shape[1:]  # (height, width, channels)

        # 結果を格納する辞書
        results = {
            "model_name": formatted_model_name,
            "model_type": model_type,
            "epoch": epochs,
            "parameter_count": int(parameter_count),
            "inferences": []
        }

        # 不良品画像に対して推論を実行
        if os.path.exists(DAMAGED_IMAGES_DIR):
            damaged_results = benchmark_images(model, DAMAGED_IMAGES_DIR, "bad", crop_box, work_name, epochs, input_shape, formatted_model_name)
            results["inferences"].extend(damaged_results)
        else:
            print(f"警告: 不良品画像ディレクトリ '{DAMAGED_IMAGES_DIR}' が存在しません。")

        # 良品画像に対して推論を実行
        if os.path.exists(PASSED_IMAGES_DIR):
            passed_results = benchmark_images(model, PASSED_IMAGES_DIR, "good", crop_box, work_name, epochs, input_shape, formatted_model_name)
            results["inferences"].extend(passed_results)
        else:
            print(f"警告: 良品画像ディレクトリ '{PASSED_IMAGES_DIR}' が存在しません。")

        # 結果を格納するディレクトリ（results/XXXX/YYYY）
        model_results_dir = os.path.join(work_results_dir, model_type)
        if not os.path.exists(model_results_dir):
            os.makedirs(model_results_dir)

        # 各モデルの結果を指定のJSONファイル名で保存
        output_result_filename = f"{work_name}_{model_type}_{str(epochs).zfill(3)}.json"
        output_result_file = os.path.join(model_results_dir, output_result_filename)
        try:
            with open(output_result_file, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)
            print(f"モデル '{formatted_model_name}' の推論結果を '{output_result_file}' に保存しました。")
        except Exception as e:
            print(f"エラー: モデル '{formatted_model_name}' の結果保存に失敗しました。詳細: {e}")

    print("\n全モデルの推論が完了しました。")

if __name__ == "__main__":
    main()
