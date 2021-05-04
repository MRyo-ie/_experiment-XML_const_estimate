from abc import ABCMeta, abstractmethod
import pandas as pd

from ..aggregate import qini_curve
from . import ModelABC
from . import DataPPP, DataPPP_UpliftBasic



class ExperimentABC(metaclass=ABCMeta):
    """
    (Abstruct) Uplift Experiment Template.
    Uplift 検証実験 を効率的に扱えるようにするための Abstruct（抽象）クラス。

    【目的】
        ・ tarin, evaluate, test, ... の抽象化。
        ・ DeepLearning モデルとの統一。
        ・ SHAP（構造的には、ExperimentABC の拡張とみなせる）（引数：学習済みModel, Data）
        ・ 分散処理？（のための要素分析）
    
    【実験の構造】
        ExperimentABC[.train(), .evaluate()] 
            ← ( ModelABC[.fit(), .predict()],  DataPPP[.get_train(), .get_eval()] )

    【Quick Start】
        ##<<--  Model:(Logistic, RF), DataPPP:Basic  -->>##
        experiments = {
            {
                'exp_name' : 'logis_basic', 
                'dataPPP' : DataPPP_UpliftBasic(X_train, X_test, y_train, y_test, w_train, w_test), 
                'model' : LogitReg_SingleModel()
            }, {
                'exp_name' : 'rf_basic', 
                'dataPPP' : dataPPP_base, 
                'model' : RF_SingleModel()
            },
        }
        ## Experiment Train
        exp_train_base = ExpTrain_UpliftBasic()
        exp_train_base.add_all(experiments)
        # または、
        # exp_train_base.add('logis', dataPPP_base, model_logis)
        # exp_train_base.add('rf', dataPPP_base, model_rf)
        exp_train_base.exec()

        ## Experiment Evalute
        exp_eval_base = ExpEvaluate_UpliftBasic(experiments)
        exp_eval_base.exec()
        result_s_abc = exp_eval_base.get_result_dict()
    """
    def __init__(self, experiments:list=None, do_exec=False):
        """
        :param list experiments: [
                {
                    'exp_name' : exp_name,
                    'dataPPP' : dataPPP, 
                    'model' : model,
                }, 
                {
                    ... 
                },
            ]
        """
        self.experiments = []
        if experiments is not None:
            self.add_all(experiments)
        if do_exec:
            self.exec()

    def add(self, exp_name:str, dataPPP:DataPPP, model:ModelABC):
        if not issubclass(type(model), ModelABC):
            raise Exception('[Error] ModelABC を継承していないモデルが入力されました。')
        if not issubclass(type(dataPPP), DataPPP):
            raise Exception('[Error] DataPPP を継承していないモデルが入力されました。')

        self.experiments.append({
            'exp_name' : exp_name,
            'dataPPP' : dataPPP, 
            'model' : model,
        })
        
    def add_all(self, experiments:list):
        for exp in experiments:
            self.add(exp['exp_name'], exp['dataPPP'], exp['model'])
            # self.add(exp['exp_name'], exp['model'], exp['dataPPP'])  # => [Error]

    @abstractmethod
    def exec(self):
        """
        「実験」を実装する。

        【実装例】 Train
            for exp in self.experiments:
                exp_name = exp['exp_name']
                data_train = exp['data'].get_train()
                model = exp['mdoel']
                # train / eval
        """
        raise NotImplementedError()



class ExpTrain_UpliftBasic(ExperimentABC):
    """
    Train の抽象化
    """
    def exec(self):
        """
        モデルを学習（train）する。
        
        例）
            for m in models:
                self.model.fit(X, params['treat'], y)
        """
        ##<<--  train  -->>##
        for exp in self.experiments:
            # exp_name = exp['exp_name']
            data_train = exp['dataPPP'].get_train()
            model = exp['model']
            
            model.fit(data_train)


class ExpEvaluate_UpliftBasic(ExperimentABC):
    """
    Evaluate の抽象化
    """
    def __init__(self, experiments:list=None, do_exec=False):
        super().__init__(experiments)
        # qini, score を保存する dict
        self.result_dicts = {
            ### イメージ
            # 'logis' : {
            #     'qini': {'train': None, 'test': None},
            #     'score': {'train': None, 'test': None}, 
            # }
        }

    def exec(self):
        """
        モデルを評価（Evaluate）する。
        """
        ##<<--  evaluate  -->>##
        for exp in self.experiments:
            exp_name = exp['exp_name']
            data_eval = exp['dataPPP'].get_eval()
            model = exp['model']
            print(f'[Info]  ======================  exp_name : {exp_name}')
            self.result_dicts[exp_name] = self.calc_qini_score(data_eval, model)

    def calc_qini_score(self, data_eval:dict, model:ModelABC):
        result_dict = {
            'qini': {'train': None, 'test': None},
            'score': {'train': None, 'test': None},
        }
        for d_type, ds in data_eval.items():
            print(f'[Info] {d_type} の score, qini を計算中...。')
            score = model.predict(ds['X'])
            res_df = pd.DataFrame({'treat':ds['treat'], 'visit':ds['visit'], 'score': score})
            qini = qini_curve(df=res_df)
            result_dict['score'][d_type] = score
            result_dict['qini'][d_type] = qini
        return result_dict

    def get_result_dict(self, model_name:str=None) -> dict:
        """
        result_dict = {
            'qini': {'train': qini_train, 'test': qini_test},
            'score': {'train': score_train, 'test': score_test},
        }
        を返す。
        （model_name が未指定の場合は、全モデルの結果を返す。）
        """
        if model_name is None:
            return self.result_dicts
        else:
            return self.result_dicts[model_name]



