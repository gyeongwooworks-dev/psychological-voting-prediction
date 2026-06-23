import warnings

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from preprocess import TARGET

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = '../data'


def find_best_weights(oof_m1, oof_m2, oof_nn3, y_true, train_df):
    """Optuna로 3개 모델 OOF 기반 최적 앙상블 가중치 탐색"""
    nn_filter_mask = train_df['familysize'] <= 50
    nn_full_oof    = np.full(len(y_true), np.nan)
    nn_full_oof[nn_filter_mask.values] = oof_nn3

    valid_mask = ~np.isnan(nn_full_oof)
    y_sub  = y_true[valid_mask]
    o1_sub = oof_m1[valid_mask]
    o2_sub = oof_m2[valid_mask]
    o3_sub = nn_full_oof[valid_mask]

    def objective(trial):
        w1 = trial.suggest_float('w1', 0.0, 1.0)
        w2 = trial.suggest_float('w2', 0.0, 1.0)
        w3 = trial.suggest_float('w3', 0.0, 1.0)
        total = w1 + w2 + w3
        if total == 0:
            return 0.0
        blend = (w1 * o1_sub + w2 * o2_sub + w3 * o3_sub) / total
        return roc_auc_score(y_sub, blend)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))
    study.optimize(objective, n_trials=300, show_progress_bar=False)

    wbest   = study.best_params
    w_total = wbest['w1'] + wbest['w2'] + wbest['w3']
    W1, W2, W3 = wbest['w1'] / w_total, wbest['w2'] / w_total, wbest['w3'] / w_total

    print(f'최적 앙상블 가중치 — M1: {W1:.4f}  M2: {W2:.4f}  M3: {W3:.4f}')
    print(f'앙상블 OOF AUC: {study.best_value:.5f}')

    return W1, W2, W3, study.best_value


def make_submission(pred_m1, pred_m2, pred_m3, W1, W2, W3,
                    oof_m1_auc, oof_m2_auc, oof_m3_auc, ensemble_auc,
                    data_dir=DATA_DIR):
    """최종 앙상블 예측 생성 및 submission.csv 저장"""
    test_idx   = pd.read_csv(f'{data_dir}/test.csv')['index']
    pred_final = W1 * pred_m1 + W2 * pred_m2 + W3 * pred_m3

    submission = pd.DataFrame({'index': test_idx, TARGET: pred_final})
    out_path   = '../submission/submission.csv'
    submission.to_csv(out_path, index=False)

    print('\n=== 모델별 OOF AUC 요약 ===')
    print(f'  Model1 (Voting)   : {oof_m1_auc:.5f}')
    print(f'  Model2 (LGBM RFE) : {oof_m2_auc:.5f}')
    print(f'  Model3 (NN)       : {oof_m3_auc:.5f}')
    print(f'  앙상블 OOF AUC    : {ensemble_auc:.5f}')
    print(f'\n최종 제출 파일 저장 완료: {out_path}')
    print(submission.head())

    return submission


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    from train import train_model1, train_model2, train_model3

    print('=== Model 1 학습 ===')
    oof_m1, pred_m1, _, _ = train_model1()

    print('\n=== Model 2 학습 ===')
    oof_m2, pred_m2, _ = train_model2()

    print('\n=== Model 3 학습 ===')
    oof_nn3, pred_m3, train_df, _ = train_model3()

    train_raw = pd.read_csv(f'{DATA_DIR}/train.csv')
    train_raw[TARGET] = train_raw[TARGET].map({1: 0, 2: 1})
    y_true = train_raw[TARGET].values

    W1, W2, W3, ens_auc = find_best_weights(oof_m1, oof_m2, oof_nn3, y_true, train_raw)

    make_submission(
        pred_m1, pred_m2, pred_m3, W1, W2, W3,
        oof_m1_auc=roc_auc_score(y_true, oof_m1),
        oof_m2_auc=roc_auc_score(y_true, oof_m2),
        oof_m3_auc=roc_auc_score(y_true[train_raw['familysize'].le(50)], oof_nn3),
        ensemble_auc=ens_auc,
    )
