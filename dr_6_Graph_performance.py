# パラメータ数、メモリ使用量、推論時間のグラフの表示

# ライブラリのインポート
import json
import matplotlib.pyplot as plt

# クラスの定義
class GraphPerformance:
    # コンストラクタ
    def __init__(self, results_path, max_epoch):
        # 初期化
        self.results_path = results_path
        self.max_epoch = max_epoch

    # メインメソッド        ### 全体的に修正が必要 ###
    def display_graph(self):
        ### 対象の決定
        # 問答無用で全てのモデルを対象とする
        # 一応全モデルを対象とし、平均を取る
        name_model = ["advanced"]
        #name_model = ["simple", "advanced", "largeInOut"]
        cnt_images = 0

        # 一括でデータをロードしておく
        data = []
        for j in range(0, len(name_model), 1):
            data.append([])
            for target_epoch in range(1, self.max_epoch+1):
                with open(self.results_path + f"autoencoder_{name_model[j]}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
                    filedata = json.load(f)
                    data[j].append(filedata)
        
        # パラメータ数の取得
        num_params = []
        for j in range(0, len(name_model), 1):
            # これはエポック数1の時のパラメータ数
            num_params.append(data[j*self.max_epoch]["parameter_count"])

        '''
        # メモリ使用量の取得
        used_memory = []
        for j in range(0, len(name_model), 1):
            cnt_images = 0
            used_memory.append(0)
            for target_epoch in range(1, self.max_epoch+1):
                with open(self.results_path + f"autoencoder_{name_model[j]}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
                    filedata = json.load(f)
                    for i in range(len(filedata["inferences"])):
                        cnt_images += 1
                        used_memory[j] += float(filedata["inferences"][i]["inference_time"])

            used_memory[j] /= cnt_images
        '''
        
        # 推論時間の取得
        cost_time = []
        for j in range(0, len(name_model), 1):
            cnt_images = 0
            cost_time.append(0)
            for target_epoch in range(1, self.max_epoch+1):
                with open(self.results_path + f"autoencoder_{name_model[j]}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
                    filedata = json.load(f)
                    for i in range(len(filedata["inferences"])):
                        cnt_images += 1
                        cost_time[j] += float(filedata["inferences"][i]["inference_time"])

            cost_time[j] /= cnt_images

        # グラフの描画
        plt.figure()
        plt.title("Performance")
        plt.xlabel("Model")
        plt.ylabel("Performance")
        plt.grid()
        plt.bar(name_model, num_params, label="Parameter Count")
        #plt.bar(name_model, used_memory, label="Used Memory")
        plt.bar(name_model, cost_time, label="Inference Time")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # テスト
    GraphPerformance("./results/", 10).display_graph()
