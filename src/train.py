import random
import warnings
from string import ascii_lowercase

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping
from optuna.samplers import TPESampler
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier, VotingClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna

from preprocess import (ANSWERS, DELAYS, NEEDENCO, TARGET, TPS,
                         build_features_m1, build_features_m2)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR     = '../data'
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────
# Model 1 — VotingClassifier (RF + LGBM + GBM)
# ──────────────────────────────────────────────────────────────

def train_model1(data_dir=DATA_DIR):
    train = pd.read_csv(f'{data_dir}/train.csv')
    test  = pd.read_csv(f'{data_dir}/test.csv')

    train[TARGET] = train[TARGET].map({1: 0, 2: 1})
    y_train = train[TARGET]
    x_train = train.drop(TARGET, axis=1)
    dataset = [x_train, test]

    build_features_m1(dataset)

    index = test['index'].copy()
    for data in dataset:
        data.drop('index', axis=1, inplace=True)

    for col in NEEDENCO:
        le = LabelEncoder()
        le.fit(pd.concat([x_train[col].astype(str), test[col].astype(str)]))
        x_train[col] = le.transform(x_train[col].astype(str))
        test[col]    = le.transform(test[col].astype(str))

    clf1      = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf2      = LGBMClassifier(random_state=0, verbose=-1)
    clf3      = GradientBoostingClassifier(random_state=0)
    soft_vote = VotingClassifier([('rf', clf1), ('lgbm', clf2), ('gbm', clf3)], voting='soft')

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_m1 = np.zeros(len(x_train))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        soft_vote.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_m1[val_idx] = soft_vote.predict_proba(x_train.iloc[val_idx])[:, 1]
        print(f'  [M1] Fold {fold+1} AUC: {roc_auc_score(y_train.iloc[val_idx], oof_m1[val_idx]):.5f}')

    print(f'Model1 OOF AUC: {roc_auc_score(y_train, oof_m1):.5f}')

    soft_vote.fit(x_train, y_train)
    pred_m1 = soft_vote.predict_proba(test)[:, 1]
    pd.DataFrame({'index': index, TARGET: pred_m1}).to_csv(f'{data_dir}/model1.csv', index=False)

    return oof_m1, pred_m1, soft_vote, x_train


# ──────────────────────────────────────────────────────────────
# Model 2 — LGBM: Optuna 튜닝 + RFE × 4 앙상블
# ──────────────────────────────────────────────────────────────

def train_model2(data_dir=DATA_DIR):
    train = pd.read_csv(f'{data_dir}/train.csv')
    test  = pd.read_csv(f'{data_dir}/test.csv')

    train[TARGET] = train[TARGET].map({1: 0, 2: 1})
    y_train = train[TARGET]
    x_train = train.drop(TARGET, axis=1)
    dataset = [x_train, test]

    build_features_m2(dataset)

    index = test['index'].copy()
    for data in dataset:
        data.drop('index', axis=1, inplace=True)

    for col in NEEDENCO:
        le = LabelEncoder()
        le.fit(pd.concat([x_train[col].astype(str), test[col].astype(str)]))
        x_train[col] = le.transform(x_train[col].astype(str))
        test[col]    = le.transform(test[col].astype(str))

    for data in [x_train, test]:
        data['Es_gender']  = data['Es']  * data['gender']
        data['Con_gender'] = data['Con'] * data['gender']
        data['Op_gender']  = data['Op']  * data['gender']

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators'     : trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate'    : trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves', 20, 150),
            'max_depth'        : trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state'     : RANDOM_STATE,
        }
        cv, scores = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), []
        for tr_idx, val_idx in cv.split(x_train, y_train):
            model = LGBMClassifier(**params)
            model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx],
                      eval_set=[(x_train.iloc[val_idx], y_train.iloc[val_idx])],
                      callbacks=[early_stopping(50, verbose=False)])
            scores.append(roc_auc_score(y_train.iloc[val_idx],
                                        model.predict_proba(x_train.iloc[val_idx])[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = {**study.best_params,
                   'objective': 'binary', 'verbosity': -1,
                   'boosting_type': 'gbdt', 'random_state': RANDOM_STATE}
    print(f'Optuna 최적 CV AUC: {study.best_value:.5f}')

    def lgbm_rfe(x_data, y_data, params, ratio=0.9, min_feats=40):
        feats, archive = x_data.columns.tolist(), []
        while True:
            model = LGBMClassifier(**params)
            X_tr, X_val, y_tr, y_val = train_test_split(
                x_data[feats], y_data, random_state=params.get('random_state', RANDOM_STATE))
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(100, verbose=False)])
            score   = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            archive.append({'model': model, 'n_feats': len(feats), 'feats': feats, 'score': score})
            print(f'  피처수={len(feats):4d}  val AUC={score:.5f}')
            next_n = int(len(feats) * ratio)
            if next_n < min_feats:
                break
            feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
            feats    = feat_imp.iloc[:next_n].index.tolist()
        return pd.DataFrame(archive)

    archives, rfe_seeds = {}, [4040, 1234, 99087, RANDOM_STATE]
    for rs in rfe_seeds:
        print(f'\n[RFE seed={rs}]')
        archives[rs] = lgbm_rfe(x_train, y_train, {**best_params, 'random_state': rs})

    def fit_predict(archive, x_tr, y_tr, x_te, params):
        best_idx = archive['score'].idxmax()
        feats    = archive.iloc[best_idx]['feats']
        model    = LGBMClassifier(**params)
        model.fit(x_tr[feats], y_tr)
        return model.predict_proba(x_te[feats])[:, 1], model, feats

    rfe_results = {}
    for rs in rfe_seeds:
        pred, model, feats = fit_predict(archives[rs], x_train, y_train, test,
                                         {**best_params, 'random_state': rs})
        rfe_results[rs] = {'pred': pred, 'model': model, 'feats': feats}

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_m2 = np.zeros(len(x_train))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        fold_pred = np.zeros(len(val_idx))
        for rs, res in rfe_results.items():
            feats = res['feats']
            m = LGBMClassifier(**{**best_params, 'random_state': rs})
            m.fit(x_train[feats].iloc[tr_idx], y_train.iloc[tr_idx])
            fold_pred += m.predict_proba(x_train[feats].iloc[val_idx])[:, 1] / len(rfe_seeds)
        oof_m2[val_idx] = fold_pred
        print(f'  [M2] Fold {fold+1} AUC: {roc_auc_score(y_train.iloc[val_idx], fold_pred):.5f}')

    print(f'Model2 OOF AUC: {roc_auc_score(y_train, oof_m2):.5f}')

    pred_all = np.mean([res['pred'] for res in rfe_results.values()], axis=0)
    pd.DataFrame({'index': index, TARGET: pred_all}).to_csv(f'{data_dir}/model2.csv', index=False)

    return oof_m2, pred_all, rfe_results


# ──────────────────────────────────────────────────────────────
# Model 3 — Neural Network (ResidualBlock + LabelSmoothing + SWA)
# ──────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ImprovedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.BatchNorm1d(input_dim), nn.Dropout(0.05),
            nn.Linear(input_dim, 256, bias=False), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
        )
        self.res1  = ResidualBlock(256, dropout=0.3)
        self.down1 = nn.Sequential(
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
        )
        self.res2  = ResidualBlock(128, dropout=0.25)
        self.down2 = nn.Sequential(
            nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
        )
        self.head  = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.down2(self.res2(self.down1(self.res1(self.stem(x))))))


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing  = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets_s, pos_weight=self.pos_weight)


def train_model3(data_dir=DATA_DIR):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'디바이스: {DEVICE}')

    DROP_COLS   = DELAYS + ['index', 'hand']
    NN_CAT_COLS = ['education', 'engnat', 'married', 'urban']

    train_data = pd.read_csv(f'{data_dir}/train.csv')
    test_data  = pd.read_csv(f'{data_dir}/test.csv')

    train_data[TARGET] = train_data[TARGET].map({1: 0, 2: 1})
    train_data = train_data.drop(
        train_data[train_data.familysize > 50].index).reset_index(drop=True)

    train_y_raw = train_data[TARGET]
    train_x_raw = train_data.drop(DROP_COLS + [TARGET], axis=1)
    test_x_raw  = test_data.drop(DROP_COLS, axis=1)

    for col in NN_CAT_COLS:
        train_x_raw[col] = train_x_raw[col].astype(str)
        test_x_raw[col]  = test_x_raw[col].astype(str)

    n_train  = len(train_x_raw)
    combined = pd.get_dummies(pd.concat([train_x_raw, test_x_raw], axis=0, ignore_index=True))
    train_x_df = combined.iloc[:n_train].reset_index(drop=True)
    test_x_df  = combined.iloc[n_train:].reset_index(drop=True)
    train_x_df, test_x_df = train_x_df.align(test_x_df, join='left', axis=1, fill_value=0)

    col_list = list(train_x_df.columns)
    qa_idx   = [col_list.index(c) for c in col_list if len(c) == 3 and c[0] == 'Q' and c[2] == 'A']
    fs_idx   = col_list.index('familysize') if 'familysize' in col_list else None
    tp_idx   = [col_list.index(c) for c in [f'tp{str(i).zfill(2)}' for i in range(1, 11)] if c in col_list]

    train_x = train_x_df.to_numpy(dtype=np.float32)
    test_x  = test_x_df.to_numpy(dtype=np.float32)

    if qa_idx:
        train_x[:, qa_idx] = (train_x[:, qa_idx] - 3.) / 2.
        test_x[:, qa_idx]  = (test_x[:, qa_idx]  - 3.) / 2.
    if fs_idx is not None:
        train_x[:, fs_idx] = (train_x[:, fs_idx] - 5.) / 4.
        test_x[:, fs_idx]  = (test_x[:, fs_idx]  - 5.) / 4.
    if tp_idx:
        train_x[:, tp_idx] = (train_x[:, tp_idx] - 3.5) / 3.5
        test_x[:, tp_idx]  = (test_x[:, tp_idx]  - 3.5) / 3.5

    train_y   = train_y_raw.to_numpy().astype(np.float32)
    INPUT_DIM = train_x.shape[1]

    train_x_t = torch.tensor(train_x, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.float32)
    test_x_t  = torch.tensor(test_x,  dtype=torch.float32)
    test_len  = len(test_x_t)

    N_REPEAT, N_SKFOLD, N_EPOCH, BATCH_SIZE = 5, 7, 60, 128
    LOADER_PARAM = {'batch_size': BATCH_SIZE, 'num_workers': 0, 'pin_memory': DEVICE != 'cpu'}

    prediction = np.zeros((test_len, 1), dtype=np.float32)
    oof_nn3    = np.zeros(len(train_y))

    for repeat in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SKFOLD, random_state=repeat, shuffle=True)
        tot = 0.
        for skfold, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
            train_loader = DataLoader(
                TensorDataset(train_x_t[train_idx], train_y_t[train_idx]),
                shuffle=True, drop_last=True, **LOADER_PARAM)
            valid_loader = DataLoader(
                TensorDataset(train_x_t[valid_idx], train_y_t[valid_idx]),
                shuffle=False, drop_last=False, **LOADER_PARAM)
            test_loader  = DataLoader(
                TensorDataset(test_x_t, torch.zeros(test_len, dtype=torch.float32)),
                shuffle=False, drop_last=False, **LOADER_PARAM)

            model     = ImprovedNN(INPUT_DIM).to(DEVICE)
            criterion = LabelSmoothingBCE(
                smoothing=0.05, pos_weight=torch.tensor([1.20665], device=DEVICE))
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=5e-3, steps_per_epoch=len(train_loader),
                epochs=N_EPOCH, pct_start=0.3)
            swa_model     = optim.swa_utils.AveragedModel(model)
            swa_start     = int(N_EPOCH * 0.75)
            swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=1e-4, anneal_epochs=5)

            prediction_t   = np.zeros((test_len, 1), dtype=np.float32)
            best_auc       = 0.
            best_val_probs = np.zeros(len(valid_idx))

            for epoch in tqdm(range(N_EPOCH), desc=f'R{repeat+1} S{skfold+1:02d}/{N_SKFOLD}'):
                model.train()
                for xx, yy in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xx.to(DEVICE)).squeeze(), yy.to(DEVICE))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if epoch < swa_start:
                        scheduler.step()

                if epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                eval_model = swa_model if epoch >= swa_start else model
                eval_model.eval()
                val_probs = []
                with torch.no_grad():
                    for xx, _ in valid_loader:
                        val_probs.append(torch.sigmoid(eval_model(xx.to(DEVICE)).squeeze()).cpu().numpy())

                val_probs_cat = np.concatenate(val_probs)
                val_auc       = roc_auc_score(train_y[valid_idx], val_probs_cat)

                if val_auc > best_auc:
                    best_auc       = val_auc
                    best_val_probs = val_probs_cat
                    tmp_preds = []
                    with torch.no_grad():
                        for xx, _ in test_loader:
                            tmp_preds.append(torch.sigmoid(eval_model(xx.to(DEVICE))).cpu().numpy())
                    prediction_t = np.vstack(tmp_preds)

            optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
            prediction[:, :] += prediction_t[:, :] / (N_REPEAT * N_SKFOLD)
            oof_nn3[valid_idx] += best_val_probs / N_REPEAT
            tot += best_auc

        print(f'Repeat {repeat+1} → avg best AUC: {tot/N_SKFOLD:.4f}')

    print(f'Model3 (NN) OOF AUC: {roc_auc_score(train_y, oof_nn3):.5f}')

    test_index = pd.read_csv(f'{data_dir}/test.csv')['index']
    pd.DataFrame({'index': test_index, TARGET: prediction.squeeze()}).to_csv(
        f'{data_dir}/model3.csv', index=False)

    return oof_nn3, prediction.squeeze(), train_data, train_x_df


if __name__ == '__main__':
    print('=== Model 1 학습 ===')
    oof_m1, pred_m1, soft_vote, x_train_m1 = train_model1()

    print('\n=== Model 2 학습 ===')
    oof_m2, pred_all, rfe_results = train_model2()

    print('\n=== Model 3 학습 ===')
    oof_nn3, pred_m3, train_nn, train_x_df = train_model3()
