# 閾値とエポック数の精度表の表示

# ライブラリのインポート
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# クラス定義
class dr_1_Table:
    # コンストラクタ
    def __init__(self, results_path):
        # 初期化
        self.results_path = results_path

    # メインメソッド
    def display_table(self):
        ### 対象の決定
        # 取得対象を決定
        target_model = self.get_target_model()
        target_epoch = self.get_target_epoch()
        target_threshold = self.get_target_threshold()
        # 数値に変換
        target_epoch = target_epoch.split(",")
        target_epoch = [int(i) for i in target_epoch]
        target_threshold = target_threshold.split(",")
        target_threshold = [float(i) for i in target_threshold]

        ### jsonデータの読み込み
        # モデル名の決定
        name_model = ""
        if target_model == "1":
            name_model = "simple"
        elif target_model == "2":
            name_model = "advanced"
        elif target_model == "3":
            name_model = "largeInOut"
        else:
            # エラーメッセージ
            print("エラー: 1~3の数字を入力してください。")
            return
        
        # jsonデータの読み込み
        result = self.read_json(name_model, target_epoch)

        # データの整形
        matrix_y = -int(-((target_threshold[1])-(target_threshold[0]))//(target_threshold[2]))
        matrix_x = -int(-((target_epoch[1])-(target_epoch[0]))//(target_epoch[2]))
        accuracy_matrix = [[0] * matrix_x for _ in range(matrix_y)]
        # 閾値ごとの正答率を取得
        for i in range(matrix_x):  # 読み込んだモデルの数だけ取得
            for j in range(matrix_y):  # 閾値の数だけ取得
                for k in range(len(result[i]["inferences"])):  # 推論結果の数だけ取得
                    #resultの中のinferenceのk番目のlabelがgoodなら
                    if result[i]["inferences"][k]["label"] == "good":
                        # 閾値を超えていたら正答をインクリメント
                        if result[i]["inferences"][k]["accuracy"] <= target_threshold[0] + j*target_threshold[2]:
                            accuracy_matrix[j][i] += 1.0
                    #resultの中のinferenceのk番目のlabelがbadなら
                    elif result[i]["inferences"][k]["label"] == "bad":
                        # 閾値を下回っていたら正答をインクリメント
                        if result[i]["inferences"][k]["accuracy"] > target_threshold[0] + j*target_threshold[2]:
                            accuracy_matrix[j][i] += 1.0
        
        # 正答率の計算
        crrct_rate = np.array(accuracy_matrix) / len(result[0]["inferences"])

        ### 結果の表示
        # 表を作成
        plt.figure(figsize=(10, 8))
        color = ["darkblue", "blue", "green", "yellowgreen", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("Accuracy_cmap", color)
        plt.imshow(crrct_rate, cmap=cmap, aspect='auto', origin='lower')

        # 軸ラベルの設定
        epoch_labels = [f'Epoch {i}' for i in range(target_epoch[0], target_epoch[1] + 1, target_epoch[2])]
        threshold_labels = [f'Thresh {target_threshold[0] + j * target_threshold[2]:.3f}' for j in range(matrix_y)]

        # xticks と yticks の数とラベルを一致させる
        plt.xticks(ticks=np.linspace(0, matrix_x - 1, len(epoch_labels)), labels=epoch_labels, rotation=45)
        plt.yticks(ticks=np.linspace(0, matrix_y - 1, len(threshold_labels)), labels=threshold_labels)

        # 各セルに正答率を百分率で表示
        for i in range(matrix_y):
            for j in range(matrix_x):
                plt.text(j, i, f'{crrct_rate[i, j] * 100:.1f}', ha='center', va='center', color='black')

        # タイトルとラベル
        plt.title(f'Accuracy Table for Model: {name_model}')
        plt.xlabel('Epochs')
        plt.ylabel('Thresholds')

        # 表の表示
        plt.tight_layout()
        plt.show()

    # モデルの選択
    def get_target_model(self):
        print("モデルを指定してください\n"
              "--1: Simple\n"
              "--2: Advanced\n"
              "--3: largeInOut\n")
        target_model = input("-> ")
        return target_model
    
    # エポック数の選択
    def get_target_epoch(self):
        print("エポック数を指定してください\n"
              "(開始,終了,刻み)\n"
              "範囲:\n"
              "-開始: 1~99\n"
              "-終了: 2~100\n"
              "-刻み: 1~100\n"
              " 例 : 1,10,1\n")
        target_epoch = input("-> ")
        return target_epoch
    
    # 閾値の選択
    def get_target_threshold(self):
        print("閾値を指定してください\n"
              "(開始,終了,刻み) 推奨範囲: [0, 0.01, 0.001]\n"
              "範囲:\n"
              "-開始: 0~0.999\n"
              "-終了: 0.001~1.0\n"
              "-刻み: 0.001~0.1\n"
              " 例 : 0.1,0.9,0.1\n")
        target_threshold = input("-> ")
        return target_threshold
    
    # jsonデータの読み込み
    def read_json(self, name_model, target_epoch):
        # jsonデータの読み込み
        result = []
        for i in range(int(target_epoch[0]), int(target_epoch[1])+1, int(target_epoch[2])):
            with open(self.results_path + "autoencoder_" + name_model + "_epochs_" + str(i).zfill(2) + "_results.json", "r") as f:
                result.append(json.load(f))
        return result
    

if __name__ == "__main__":
    #テスト
    dr_1_Table("./results/").display_table()
