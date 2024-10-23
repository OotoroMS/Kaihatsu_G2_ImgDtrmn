# 推論時間のグラフの表示

# ライブラリのインポート
import matplotlib.pyplot as plt
import json

# クラスの定義
class GraphInferenceTime:
    # コンストラクタ
    def __init__(self, results_path):
        # 初期化
        self.results_path = results_path

    def show_graph(self):
        ### 対象の決定
        # 問答無用で全てのモデルを対象とする
        # 一応全モデルの推論時間を取得
        cnt_images = 0
        cost_time = []
        name_model = ["simple", "advanced", "largeInOut"] # 全データがそろったら置き換え

        for j in range(0, len(name_model), 1):
            cnt_images = 0
            cost_time.append(0)
            for target_epoch in range(1, 101):
                with open(self.results_path + f"autoencoder_{name_model[j]}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
                    filedata = json.load(f)
                    for i in range(len(filedata["inferences"])):
                        cnt_images += 1
                        cost_time[j] += float(filedata["inferences"][i]["inference_time"])

            cost_time[j] /= cnt_images # 平均値を取得
            cost_time[j] *= 1000 # ミリ秒に変換
        
        ### グラフの表示
        # グラフの描画
        plt.figure()
        plt.title("Inference Time")
        plt.xlabel("Model")
        plt.ylabel("Average Inference Time [ms]")
        plt.grid()
        plt.bar(["simple", "advanced", "largeInOut"], cost_time)
        plt.show()


if __name__ == "__main__":
    # テスト
    GraphInferenceTime("./results/").show_graph()

