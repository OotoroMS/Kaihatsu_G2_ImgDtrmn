# メモリ使用量のグラフの表示

# ライブラリのインポート
import json
import matplotlib.pyplot as plt

# クラスの定義
class GraphMemory:
    # コンストラクタ
    def __init__(self, results_path, target_epoch=1):
        # 初期化
        self.results_path = results_path
        self.target_epoch = target_epoch

    # メインメソッド
    def display_graph(self, target_epoch):
        ### 対象の決定
        # 問答無用で全てのモデルを対象とする
        # 代表としてエポック数が最小のものを選択
        name_model = ["simple", "advanced", "largeInOut"]
        used_memory = []
        for i in range(1, 4):
            with open(self.results_path + f"autoencoder_{name_model[i]}_epochs_{str(target_epoch).zfill(2)}_results.json", "r") as f:
                #used_memory.append(json.load(f)["used_memory"]) # メモリ使用量の取得 (未実装)
                pass
        
        ### グラフの表示
        # グラフの描画
        plt.figure()
        plt.title("Memory Usage")
        plt.xlabel("Model")
        plt.ylabel("Memory Usage")
        plt.grid()
        plt.bar(name_model, used_memory)
        plt.show()

if __name__ == "__main__":
    # テスト
    GraphMemory("./results/").display_graph(1)
