import numpy as np
import pandas as pd
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=1):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if len(feature_vector) != len(target_vector):
        raise ValueError("feature_vector and target_vector have different sizes!")
    N = len(feature_vector)
    feature_vector_sort = np.array(sorted(feature_vector).copy())

    # сдвинем список на один вправо и после посчитаем среднее соседних значений
    # здесь преобразую в множество, а потом обратно в список, чтобы избавиться от дубликатов
    thresholds = pd.DataFrame(
        np.array(
            sorted(list(set((feature_vector_sort[1:] + feature_vector_sort[:-1]) / 2)))
        ).reshape(1, -1)
    )
    if thresholds.shape[1] == 0:
        return None, None, None, None
    thresholds_df = pd.concat([thresholds] * N).reset_index(drop=True)

    data = pd.DataFrame(
        {"original": feature_vector, "target": target_vector}
    ).reset_index(drop=True)
    df = pd.concat([thresholds_df, data], axis=1)
    thresholds_columns = list(range(thresholds.shape[1]))
    left_right = df.apply(
        lambda row: row[thresholds_columns] <= row["original"], axis=1
    ).astype(int)
    left_right_target = pd.concat([left_right, df["target"]], axis=1)

    total = left_right_target.drop(columns=["target"]).count().values
    left_total = left_right_target.drop(columns=["target"]).sum().values
    right_total = total - left_total

    class1, class0 = (
        left_right_target[left_right_target.target == 1],
        left_right_target[left_right_target.target == 0],
    )

    total1 = class1.drop(columns=["target"]).count().values
    left1 = class1.drop(columns=["target"]).sum().values
    right1 = total1 - left1

    total0 = class0.drop(columns=["target"]).count().values
    left0 = class0.drop(columns=["target"]).sum().values
    right0 = total0 - left0

    result = pd.DataFrame(
        {
            "thresholds": thresholds.values[0],
            "left1": left1,
            "left0": left0,
            "left_total": left_total,
            "right1": right1,
            "right0": right0,
            "right_total": right_total,
            "total": left_total + right_total,
        }
    ).dropna()

    # убираем пороги с пустыми подмножествами
    result = result[
        (result["left_total"] >= min_samples_leaf)
        & (result["right_total"] >= min_samples_leaf)
    ]

    # доли объектов класса 1 и 0
    result["left_p1"] = result["left1"] / result["left_total"]
    result["left_p0"] = result["left0"] / result["left_total"]
    result["right_p1"] = result["right1"] / result["right_total"]
    result["right_p0"] = result["right0"] / result["right_total"]

    # H(R)
    result["H_left"] = 1 - result["left_p1"] ** 2 - result["left_p0"] ** 2
    result["H_right"] = 1 - result["right_p1"] ** 2 - result["right_p0"] ** 2

    # Q(R)
    result["gini"] = (
        -result["left_total"] / result["total"] * result["H_left"]
        - result["right_total"] / result["total"] * result["H_right"]
    )
    gini_best = result["gini"].max()
    threshold_best = result[result["gini"] == gini_best]["thresholds"].min()
    return result["thresholds"].values, result["gini"].values, threshold_best, gini_best


class DecisionTree:
    def __init__(
        self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1
    ):
        if np.any(
            list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._depth = 0
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            self._depth = max(depth, self._depth)
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = sub_X.iloc[:, feature].values
            elif feature_type == "categorical":
                counts = Counter(sub_X.iloc[:, feature])
                clicks = Counter(sub_X.iloc[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_count / current_click
                    else:
                        current_click = 0
                        ratio[key] = 0
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))
                )
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories))))
                )
                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], sub_X.iloc[:, feature]))
                )
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(
                feature_vector, sub_y, self._min_samples_leaf
            )
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector <= threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(
                        map(
                            lambda x: x[0],
                            filter(lambda x: x[1] < threshold, categories_map.items()),
                        )
                    )
                else:
                    raise ValueError

        if (
            feature_best is None
            or depth == self._max_depth
            or gini_best is np.nan
            or len(sub_y[split]) < self._min_samples_split
            or len(sub_y[np.logical_not(split)]) < self._min_samples_split
        ):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            self._depth = max(depth, self._depth)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(
            pd.DataFrame(sub_X[split]), sub_y[split], node["left_child"], depth + 1
        )
        self._fit_node(
            pd.DataFrame(sub_X[np.logical_not(split)]),
            sub_y[np.logical_not(split)],
            node["right_child"],
            depth + 1,
        )

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ *вжух*
        if x is None or node == {}:
            return None
        while True:
            if node["type"] == "nonterminal":
                feature_ind = node["feature_split"]
                if "threshold" in node:  # real
                    threshold = node["threshold"]
                    if x.iloc[feature_ind] <= threshold:
                        node = node["left_child"]
                    else:
                        node = node["right_child"]
                else:  # categorical
                    categories_split = node["categories_split"]
                    if x.iloc[feature_ind] in categories_split:
                        node = node["left_child"]
                    else:
                        node = node["right_child"]
            else:
                return node["class"]

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = y.values.reshape(1, -1)[0]
        self._fit_node(X, y, self._tree, self._depth)

    def predict(self, X):
        X = pd.DataFrame(X)
        predicted = []
        for _, x in X.iterrows():
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
