import os
import json
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

# ディレクトリの設定
#MODEL_DIR = "./model"
MODEL_DIR = "./model_only"
DAMAGED_IMAGES_DIR = "./DamagedImages"  # 不良品画像のディレクトリ
PASSED_IMAGES_DIR = "./PassedImages"    # 良品画像のディレクトリ
RESULTS_DIR = "./results"
OUTPUT_IMAGES_DIR = "./output_images"   # 出力画像を保存するディレクトリ

# クロップ範囲の設定 (left, upper, right, lower)
crop_box = (300, 720, 300+2112, 720+784)

# ディレクトリが存在しない場合は作成
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(OUTPUT_IMAGES_DIR):
    os.makedirs(OUTPUT_IMAGES_DIR)

def load_model_and_info(model_path):
    """モデルを読み込み、パラメータ数とメモリ使用量を取得する関数"""
    model = load_model(model_path)
    model_type = "simple" if "simple" in model_path else "advanced" if "advanced" in model_path else "largeInOut"
    epochs = int(model_path.split('_')[-1].split('.')[0])  # モデル名からエポック数を抽出
    parameter_count = model.count_params()
    
    return model, model_type, epochs, parameter_count

def preprocess_image(image_path):
    """画像を読み込み、前処理を行う関数"""
    image = load_img(image_path, color_mode='grayscale')
    image = image.crop(crop_box)  # 画像サイズをリサイズ
    image_array = img_to_array(image) / 255.0  # 正規化
    image_array = np.expand_dims(image_array, axis=0)  # バッチ次元を追加
    return image_array

def save_output_image(output_image, output_path):
    """出力画像を保存する関数"""
    # 画像を正規化から元に戻して保存
    output_image = np.squeeze(output_image, axis=0)  # バッチ次元を削除
    output_image = output_image * 255.0  # 0-1の範囲を0-255に戻す
    output_image = output_image.astype(np.uint8)  # uint8形式に変換
    output_img = array_to_img(output_image)
    output_img.save(output_path)

def inference(model, image, image_name):
    """推論を行い、精度と推論時間を返す関数"""
    start_time = time.time()
    output = model.predict(image)
    end_time = time.time()

    # 出力画像を保存するパス
    output_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"output_{image_name}")
    save_output_image(output, output_image_path)
    
    # MSE（平均二乗誤差）を精度の指標として使用
    mse = np.mean((output - image) ** 2)
    inference_time = end_time - start_time
    
    return float(mse), float(inference_time), output_image_path  # float32をfloatに変換

def benchmark_images(model, image_dir, label):
    """指定されたディレクトリの画像を使用して推論を実行し、結果を収集する関数"""
    results = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_name)
        image = preprocess_image(image_path)
        accuracy, inference_time, output_image_path = inference(model, image, image_name)
        
        # 推論結果を追加
        results.append({
            "image_path": image_path,
            "accuracy": accuracy,
            "inference_time": inference_time,
            "label": label,  # "good" または "bad" のラベル
            "output_image_path": output_image_path  # 出力画像の保存先
        })
    return results

def main():
    # モデルディレクトリからモデルをロード
    for model_name in os.listdir(MODEL_DIR):
        if model_name.endswith(".keras"):  # .kerasファイルのみを対象
            model_path = os.path.join(MODEL_DIR, model_name)
            model, model_type, epochs, parameter_count = load_model_and_info(model_path)
            
            # 結果を格納するリスト
            results = {
                "model_name": model_name,
                "model_type": model_type,
                "epoch": epochs,
                "parameter_count": int(parameter_count),  # もしもパラメータ数がint以外であれば変換
                "inferences": []
            }

            # 不良品画像に対して推論を実行
            damaged_results = benchmark_images(model, DAMAGED_IMAGES_DIR, "bad")
            results["inferences"].extend(damaged_results)

            # 良品画像に対して推論を実行
            passed_results = benchmark_images(model, PASSED_IMAGES_DIR, "good")
            results["inferences"].extend(passed_results)
            
            # JSONファイルに結果を保存
            output_path = os.path.join(RESULTS_DIR, f"{model_name.split('.')[0]}_results.json")
            with open(output_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
