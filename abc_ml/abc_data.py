from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



class DataPPP(metaclass=ABCMeta):
    """
    (Abstruct) ML Data PreProcess Pattern.

    実験（学習 → score, qini曲線）に使用する train/test データの生成をパターン化する。
    バラバラだったので、統一しやすいように規格化した。
    
    【目標】
        datasetの前処理 を抽象化する。
            ・ Xtrain, Xtest, ytrain, ytest, wtrain, wtest に統合・分割する。
            ・ .csvファイル → pd.DataFrame
            ・ 画像ファイル → 
            ・ 文章ファイル → 
        を抽象化する。

    【実装例】
        class Data_UpliftBasic(UpliftModelTmpl):
            def _build_exp_dataset(self):
                ・・・
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.w_train = None
        self.X_test = None
        self.y_test = None
        self.w_test = None
    
    def load_csv(self, path, X_clms:list=None, y_clms:list=None, w_clms:list=None, n=None):
        # dataset for training
        # if mini:
        #     path = './data/criteo-uplift_mini.csv'
        reader = pd.read_csv(path)
        df = reader.sample(n=n).reset_index(drop=True)

        X = pd.DataFrame(df, columns=X_clms)
        w = np.array(df.treatment).astype(int)
        y = np.array(df.visit).astype(int)

        X_train, X_test, w_train, w_test, y_train, y_test = train_test_split(X.values, w, y, test_size=0.3,
                                                                             random_state=30, stratify=y)
        X_train, X_valid, w_train, w_valid, y_train, y_valid = train_test_split(X_train, w_train, y_train,
                                                                                test_size=1 / 3, random_state=30, stratify=y_train)

        X = pd.DataFrame(X_train) # 特徴データ
        treat = pd.Series(w_train) # 介入データ
        y = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        X_valid = pd.DataFrame(X_valid)
        
        X_ = pd.concat([X, X_valid], axis=0).reset_index(drop=True)
        use_variables = list(delete_variables(X_, 0.6).columns)

        # dataをディープコピーする。
        if X_clms is None:
            self.X_train = Xtrain.copy()
            self.X_test = Xtest.copy()
        else:
            self.X_train = pd.DataFrame(Xtrain, columns=X_clms)
            self.X_test = pd.DataFrame(Xtest, columns=X_clms)
        # dataが破損していないか確認する。（未）
        # self._check_dataset(self.X_train)



    def generateValidationData(self, X_scaled, w, y):
        from sklift.metrics import uplift_at_k, uplift_auc_score, qini_auc_score  # , weighted_average_uplift

        # Area Under Qini Curve
    #    tm_qini_auc = qini_auc_score(y_true=y_test, uplift=uplift_preds, treatment=w_test)
    #    print('towmodel', tm_qini_auc)

        X_train, X_test, w_train, w_test, y_train, y_test = train_test_split(X_scaled.values, w, y, test_size=0.3,
                                                                             random_state=30, stratify=y)
        X_train, X_valid, w_train, w_valid, y_train, y_valid = train_test_split(X_train, w_train, y_train,
                                                                                test_size=1 / 3, random_state=30, stratify=y_train)
        return X_train, X_valid, X_test, w_train, w_valid, w_test,  y_train, y_valid, y_test




    def CRITEO_data(self, n=100000, mini=False):
        """
        Criteo Benchmark Dataset for Uplift-Modeling

        :return: Input Pipeline
        """

        # dataset for training
        csv_file = './data/criteo-uplift.csv'
        if mini:
            csv_file = './data/criteo-uplift_mini.csv'
        reader = pd.read_csv(csv_file)
        df = reader.sample(n=n).reset_index(drop=True)

        X = df.drop(['treatment', 'conversion', 'visit', 'exposure'], axis=1)
        w = np.array(df.treatment).astype(int)
        y = np.array(df.visit).astype(int)

        X_train, X_valid, X_test, w_train, w_valid, w_test, y_train, y_valid, y_test = self.generateValidationData(X, w, y)

        X = pd.DataFrame(X_train) # 特徴データ
        treat = pd.Series(w_train) # 介入データ
        y = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        X_valid = pd.DataFrame(X_valid)
        
        X_ = pd.concat([X, X_valid], axis=0).reset_index(drop=True)
        use_variables = list(delete_variables(X_, 0.6).columns)

        return X, treat, y, X_train, X_valid, X_test, w_train, w_valid, w_test, y_train, y_valid, y_test, use_variables




class DataABC(metaclass=ABCMeta):
    """
    (Abstruct) ML Dataset Frame.

    機械学習 Experiment（train, test）に使用する train, test データの共通規格。
    ExperimentABC, ModelABC の Input の規格。
    
    【目標】
        datasetの型 を抽象化する。
        用途：
            ・ train/test および X/y/w の整理
            ・ Model や 前処理プログラムの Input 規格の統一。
                ・ ModelABC を継承する、機械学習モデル のInput。
                ・ バイアス除去（IPTW, DR, SDR） のInput。
                ・ ノイズ除去 のInput。
                ・ 異常検知（？） のInput。

    【実装例】
        class Data_UpliftBasic(UpliftModelTmpl):
            def get_train(self):
                ・・・
            def get_eval(self):
                ・・・
    """
    def __init__(self, Xtrain, Xtest, ytrain, ytest, wtrain, wtest, use_variables:list=None):
        # dataをディープコピーする。
        if use_variables is None:
            self.X_train = Xtrain.copy()
            self.X_test = Xtest.copy()
        else:
            self.X_train = pd.DataFrame(Xtrain, columns=use_variables)
            self.X_test = pd.DataFrame(Xtest, columns=use_variables)

        self.y_train = ytrain.copy()
        self.w_train = wtrain.copy()
        self.y_test = ytest.copy()
        self.w_test = wtest.copy()
        # dataが破損していないか確認する。（未）
        # self._check_dataset(self.X_train)

    @abstractmethod
    def get_train(self) -> dict:
        """
        例）
            X = self.X_train
            y = self.w_train * self.y_train + (1 - self.w_train) * (1 - self.y_train)
            return {
                'X' : X,
                'y' : y,
            }
        """
        raise NotImplementedError()

    @abstractmethod
    def get_eval(self) -> dict:
        """
        例）
            return {
                'train' : {
                    'X': self.X_train, 
                    'treat': self.w_train, 
                    'visit': self.y_train,
                },
                'test' : { 
                    'X': self.X_test, 
                    'treat': self.w_test, 
                    'visit': self.y_test,
                }
            }
        """
        raise NotImplementedError()
    
    def _check_dataset(self, data):
        pass




class Data_UpliftBasic(DataABC):
    """
    スタンダードなデータ配分。
    ・ train -> fit(X, y)
        ・ X = X_train
        ・ y = w_train * y_train + (1 - w_train) * (1 - y_train)
    ・ eval -> pred(X, treat, visit)
        ・ train (X, treat, visit)
        ・ test (X, treat, visit)
        ※ qini_curve() で指定する。
    """
    def get_train(self):
        X = self.X_train
        y = self.w_train * self.y_train + (1 - self.w_train) * (1 - self.y_train)
        return {
            'X' : X,
            'y' : y,
        }

    def get_eval(self):
        return {
            'train' : {
                'X': self.X_train, 
                'treat': self.w_train, 
                'visit':self.y_train,
            },
            'test' : { 
                'X': self.X_test, 
                'treat': self.w_test, 
                'visit':self.y_test,
            }
        }


