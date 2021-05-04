
import numpy as np
from abc import ABCMeta, abstractmethod



class ModelABC(metaclass=ABCMeta):
    """
    (Abstruct) Uplift Model Template.
    Uplift model を効率的に扱えるようにするための Abstruct（抽象）クラス。
    継承すると、それを実装しただけで、テストまで自動実行できます。

    【実装例】
        class LogisticRegressionModel(UpliftModelTmpl):
            def _init_model(self):
                self.model = LogisticRegression()
    """
    def __init__(self, model_instance):
        self.model = model_instance
        if self.model is None:
            raise Exception('[Error] self.model が　None です')
    
    '''
    # または、継承先で以下を実装する。
    def __init__(self, arg1, arg2):
        self.model = LogisticRegression(arg1, arg2)
    '''

    @abstractmethod
    def fit(self, datadict) -> dict:
        """
        モデルを学習（fit）する。

        :param dict datadict: {
                'X' : X,
                'y' : y,
            }
            という規格の json を想定。（自由に拡張可能）
            ※ key に 'X', 'y' は含むことを推奨
        【実装例】
            X = datadict.get_train()['X']
            y = datadict.get_train()['y']
            self.model.fit(X, y)
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        """
        モデルで予測（predict）する。

        :return np.ndarray : model の予測結果(array)
        【実装例】
            score_arr = 2 * self.predict_proba(data)[:,1] - 1
            return np.array(score_arr)
        """
        raise NotImplementedError()
    

    @abstractmethod
    def predict_orgf(self, data):
        """
        モデルで予測（predict）する。
        （何も手を加えない、original の出力値を return すること）
        （SHAP などに使用を想定）

        :return np.ndarray : model の予測結果(array)
        【実装例】
            res = np.array(self.model.predict_proba(data))
            res = res.reshape(data.shape)
            return res
        """
        raise NotImplementedError()



