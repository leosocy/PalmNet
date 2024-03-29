import keras
import matplotlib.pyplot as plt
import numpy as np

from palmnet.dataset import SupportedDatasetName, load_train_test_dataset
from palmnet.losses import ArcLoss

if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = load_train_test_dataset(
        [SupportedDatasetName.POLYU],
        for_predict=True,
    )

    with keras.utils.custom_object_scope({"ArcLoss": ArcLoss()}):
        model = keras.models.load_model("./palmnet.tf")
        train_features = model.predict(train_features)

        # 存储特征库
        feature_database = {"features": train_features, "labels": train_labels}

        test_features = model.predict(test_features)

        # 假设匹配度量为余弦相似度
        from scipy.spatial.distance import cosine

        correct_matches = 0
        total_matches = len(test_features)

        avg = np.mean(train_features)

        def AdjustedCosine(dataA, dataB, avg):
            sumData = np.dot((dataA - avg), (dataB - avg))
            denom = np.linalg.norm(dataA - avg) * np.linalg.norm(dataB - avg)
            return 0.5 + 0.5 * (sumData / denom)

        all_similarities = []
        for i, test_feature in enumerate(test_features):
            # 计算余弦相似度
            similarities = [
                AdjustedCosine(test_feature, train_feature, avg) for train_feature in feature_database["features"]
            ]

            all_similarities = np.append(all_similarities, np.max(similarities))
            # similarities_labels = np.column_stack((similarities, train_labels))
            # similarities_labels = np.sort(similarities_labels, axis=0)[::-1]

            max_similarity = np.max(similarities)
            mean_similarity = np.mean(similarities)
            variance_similarity = np.var(similarities)
            matched_label = feature_database["labels"][similarities.index(max_similarity)]

            # 验证匹配是否正确
            if matched_label == test_labels[i]:
                if max_similarity > 0.95:
                    print(
                        f"Matched: real label: {test_labels[i]}, matched label: {matched_label}. "
                        f"similarity: max {max_similarity} mean: {mean_similarity} var: {variance_similarity}",
                    )
                    correct_matches += 1
                else:
                    print(
                        f"Matched But Small than Threshold: real label: {test_labels[i]}, matched label: {matched_label}. "
                        f"similarity: max {max_similarity} mean: {mean_similarity} var: {variance_similarity}",
                    )
                    correct_matches += 1
            else:
                print(
                    f"NOT Matched: real label: {test_labels[i]}, matched label: {matched_label}. "
                    f"similarity: max {max_similarity} mean: {mean_similarity} var: {variance_similarity}",
                )

        accuracy = correct_matches / total_matches
        print("Accuracy:", accuracy)

        hist, bins = np.histogram(all_similarities, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个箱的中心

        # 使用matplotlib.pyplot.plot绘制概率密度曲线
        plt.plot(bin_centers, hist, label="数值分布曲线", color="blue")

        # 或者使用matplotlib.pyplot.bar绘制直方图
        # plt.bar(bin_centers, hist, width=bins[1] - bins[0], label='数值分布直方图', color='blue', alpha=0.7)

        # 可视化结果
        plt.title("数值分布")
        plt.xlabel("值")
        plt.ylabel("概率密度")
        plt.legend()
        plt.show()
