from ultralytics import YOLO


def main():

    model = YOLO("best.pt")

    results = model("test.jpg")

    # 出力ファイルを開く
    with open("result.txt", "w", encoding="utf-8") as f:

        f.write("=== YOLO検出結果 ===\n\n")

        for result in results:

            if len(result.boxes) == 0:
                f.write("何も検出されませんでした。\n")
                continue

            for box in result.boxes:

                # クラスID
                class_id = int(box.cls[0])

                # クラス名
                class_name = result.names[class_id]

                # 信頼度
                confidence = float(box.conf[0])

                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # コンソールにも表示
                print(f"名前       : {class_name}")
                print(f"信頼度     : {confidence:.2f}")
                print(f"位置       : ({x1}, {y1}) - ({x2}, {y2})")
                print("------------------------")

                # ファイルにも出力
                f.write(f"名前       : {class_name}\n")
                f.write(f"信頼度     : {confidence:.2f}\n")
                f.write(f"位置       : ({x1}, {y1}) - ({x2}, {y2})\n")
                f.write("------------------------\n")

    print("\n検出結果を result.txt に保存しました。")


if __name__ == "__main__":
    main()