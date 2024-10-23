# 良品, 不良品それぞれの正答率と閾値のグラフの表示

# ライブラリのインポート
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# クラスの定義
class Graph_crrctvsdmg:
    # コンストラクタ
    def __init__(self, results_path):
        # 初期化
        self.results_path = results_path

    # メインメソッド
    def display_graph(self):
        ### 対象の決定
        # 取得対象を決定
        target_model = self.get_target_model()
        target_epoch = self.get_target_epoch()
        target_threshold = self.get_target_threshold()
        # 数値に変換
        target_epoch = int(target_epoch)
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
        list_length = -int(-((target_threshold[1]) - (target_threshold[0])) // (target_threshold[2]))
        accuracy_list_CRRCT = [0] * list_length  # 良品の正答
        accuracy_list_DMGD = [0] * list_length   # 不良品の正答
        cnt_CRRCT = 0  # 良品の数
        cnt_DMGD = 0   # 不良品の数

        # 閾値ごとの正答率を取得
        for j in range(list_length):  # 閾値の数だけ取得
            # 初期化
            cnt_CRRCT = 0  # 良品の数
            cnt_DMGD = 0   # 不良品の数
            threshold = target_threshold[0] + j * target_threshold[2]
            for inference in result["inferences"]:  # 推論結果の数だけ取得
                if inference["label"] == "good":
                    cnt_CRRCT += 1  # 良品の数をインクリメント
                    # 閾値を超えていたら正答をインクリメント
                    if inference["accuracy"] <= threshold:
                        accuracy_list_CRRCT[j] += 1  # 良品の正答をインクリメント
                elif inference["label"] == "bad":
                    cnt_DMGD += 1  # 不良品の数をインクリメント
                    # 閾値を下回っていたら正答をインクリメント
                    if inference["accuracy"] > threshold:
                        accuracy_list_DMGD[j] += 1  # 不良品の正答をインクリメント

        # 正答率の計算
        if cnt_CRRCT > 0:
            crrct_rate = np.array(accuracy_list_CRRCT) / cnt_CRRCT
        else:
            crrct_rate = np.zeros(list_length)  # 良品のデータがない場合はゼロ配列
        if cnt_DMGD > 0:
            dmgd_rate = np.array(accuracy_list_DMGD) / cnt_DMGD
        else:
            dmgd_rate = np.zeros(list_length)  # 不良品のデータがない場合はゼロ配列

        # 閾値のリストを作成
        thresholds = np.arange(target_threshold[0], target_threshold[0] + list_length * target_threshold[2], target_threshold[2])

        ### 結果の表示
        # 折れ線グラフを2本表示
        plt.figure()
        plt.plot(thresholds, crrct_rate, color="red", label="Passed")
        plt.plot(thresholds, dmgd_rate, color="blue", label="Failed")
        plt.xlabel("Threshold")
        plt.ylabel("Correct Rate")
        plt.title("Correct Rate vs Threshold")
        plt.legend()
        plt.grid()
        plt.ylim(0, 1.05)  # 正答率は0から1の範囲に設定
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
        print("エポック数を指定してください (1~100)")
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
        result = None
        with open(self.results_path + f"autoencoder_{name_model}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
            result = json.load(f)
        return result
    
if __name__ == "__main__":
    # テスト
    Graph_crrctvsdmg("./results/").display_graph()
