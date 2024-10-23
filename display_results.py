### メインプログラム ###

# ライブラリのインポート

# 自作モジュールのインポート
import dr_1_Table
import dr_2_Graph
import dr_3_Graph_crrctvsdmg
import dr_4_Graph_memory
import dr_5_Graph_inference_time
import dr_6_Graph_performance

# グローバル変数
RESULTS_PATH = "./results/"
MAX_EPOCH = 100

# メイン関数
def main():
    # プログラムの開始メッセージ
    print("###ベンチマーク結果表示###")
    print("-手法選択\n"
          "--1: 閾値とエポック数の精度表\n"
          "  **適切な閾値を探すための表.\n"
          "--2: 閾値と正答率の折れ線グラフ\n"
          "  **適切な学習量、モデルの構造を探すためのグラフ.\n"
          "--3: 良品, 不良品それぞれの正答率と閾値のグラフ\n"
          "  **閾値の設定による正答率の変化を確認するためのグラフ.\n"
          "--4: メモリ使用量のグラフ\n"
          "  **モデルの軽量化を目指すためのグラフ.\n"
          "--5: 推論時間のグラフ\n"
          "  **推論時間の短縮を目指すためのグラフ.\n"
          #"--6: パラメータ数、メモリ使用量、推論時間のグラフ\n"
          #"  **モデルのパフォーマンスを確認するためのグラフ.\n"
          )
    print("###")
    
    # ユーザーの入力
    user_input = input("表示したい結果を選択してください(1~3)\n-> ")
    print("\n")

    # 選択された手法による結果の表示
    if user_input == "1":
        # 閾値とエポック数の精度表の表示
        print("###1. 閾値とエポック数の精度表###")
        # クラスのインスタンス化
        dr_1_Table.dr_1_Table(RESULTS_PATH).display_table()
    elif user_input == "2":
        # 閾値と正答率の折れ線グラフの表示
        print("###2. 閾値と正答率の折れ線グラフ###")
        # クラスのインスタンス化
        dr_2_Graph.dr_2_Graph(RESULTS_PATH).display_graph()
    elif user_input == "3":
        # 良品, 不良品それぞれの正答率と閾値のグラフの表示
        print("###3. 良品, 不良品それぞれの正答率と閾値のグラフ###")
        # クラスのインスタンス化
        dr_3_Graph_crrctvsdmg.Graph_crrctvsdmg(RESULTS_PATH).display_graph()
    elif user_input == "4":
        # メモリ使用量のグラフの表示
        print("###4. メモリ使用量のグラフ###")
        # クラスのインスタンス化
        dr_4_Graph_memory.GraphMemory(RESULTS_PATH).display_graph(1)
    elif user_input == "5":
        # 推論時間のグラフの表示
        print("###5. 推論時間のグラフ###")
        # クラスのインスタンス化
        dr_5_Graph_inference_time.GraphInferenceTime(RESULTS_PATH).show_graph()
    elif user_input == "6":
        # パラメータ数、メモリ使用量、推論時間のグラフの表示
        print("###6. パラメータ数、メモリ使用量、推論時間のグラフ###")
        # クラスのインスタンス化
        dr_6_Graph_performance.dr_6_Graph_performance(RESULTS_PATH).display_graph()
    else:
        # エラーメッセージ
        print("エラー: 1~3の数字を入力してください。")

    # プログラムの終了メッセージ
    print("プログラムを終了します。")

# メイン関数の実行
if __name__ == "__main__":
    main()