import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.preprocessing import StockPreprocessor

class WalkForwardBacktester(object):
    def __init__(self, agent):
        self.agent = agent

    def ema_update(self, old_w, new_w, alpha=0.1):
        if old_w is None:
            return copy.deepcopy(new_w)
        updated = {}
        for k in old_w:
            updated[k] = alpha * new_w[k].detach().cpu() + (1-alpha) * old_w[k].detach().cpu()

        return updated
    
    def predict_step(self, model, data_slice):
        device = self.agent.device
        seq_len = self.agent.seq_len
        pred_len = self.agent.pred_len
        x = torch.from_numpy(data_slice[-seq_len:]).unsqueeze(0).to(device).float()

        x_mark = torch.zeros(1, seq_len, 4, device=device)
        y_mark = torch.zeros(1, pred_len, 4, device=device)
        dec_inp = torch.zeros(1, pred_len, x.shape[-1], device=device)

        model.eval()
        with torch.no_grad():
            out = model(x, x_mark, dec_inp, y_mark)

            if isinstance(out, (tuple, list)):
                out = out[0]

            preds = out[:, -pred_len:,0].cpu().numpy()[0]
        return preds

    def run_backtest(self, cutoff_idx):
        scalers = []

        pred_len = self.agent.pred_len

        print("\n[Backtest] Training base model...")

        #train base model
        base_df = self.agent.feat_df.iloc[:cutoff_idx]
        base_prep = StockPreprocessor(feature_cols=self.agent.feature_cols, target_col = "close")

        print("Feature order:", base_prep.feature_cols)
        print("Target index:", base_prep.target_idx)
        print(base_prep.feature_cols[base_prep.target_idx])

        base_prep.fit(base_df)
        base_scaled = base_prep.transform(base_df)
        base_model,train_cfg = self.agent._init_model()
        self.agent._train_model(base_model, base_scaled, train_cfg)

        base_weights = copy.deepcopy(base_model.state_dict())

        #walk forward backtesting

        print("\n[Backtest] Walk-forward starting...")

        model, train_cfg = self.agent._init_model()
        model.load_state_dict(base_weights)

        W_ema = copy.deepcopy(base_weights)

        preds_all = []
        actual_all = []
        timeline_ix = []

        end_idx = len(self.agent.feat_df) - pred_len

        max_steps = 100
        for d in range(cutoff_idx, min(end_idx, cutoff_idx + max_steps)):
            train_df = self.agent.feat_df.iloc[:d]
            prep = StockPreprocessor(feature_cols=self.agent.feature_cols, target_col = "close")
            if d == cutoff_idx:
                print("Feature order:", prep.feature_cols)
                print("Target index:", prep.target_idx)
                print(prep.feature_cols[prep.target_idx])
            prep.fit(train_df)
            train_scaled = prep.transform(train_df)

            scalers.append(prep)
            #warm start training
            self.agent._train_model(model, train_scaled, train_cfg, fine_tune=True)

            #prediction
            preds = self.predict_step(model, train_scaled)

            #actual
            actual = self.agent.feat_df["close"].values[d:d+pred_len]

            preds_all.append(preds)
            actual_all.append(actual)
            timeline_ix.append(d)

            W_ema = self.ema_update(W_ema, model.state_dict())

            print(f"[Backtest] step {d}/{end_idx}")

        preds_all = np.array(preds_all)
        actual_all= np.array(actual_all)

        #final training
        print("\n[Backtest] Final training on full data...")

        full_prep = StockPreprocessor(feature_cols=self.agent.feature_cols, target_col="close")
        full_prep.fit(self.agent.feat_df)
        full_scaled = full_prep.transform(self.agent.feat_df)

        final_model,train_cfg = self.agent._init_model()
        final_model.load_state_dict(W_ema)
        final_model.to(self.agent.device)
        self.agent._train_model(final_model, full_scaled, train_cfg, fine_tune=False)

        final_preds = self.predict_step(final_model, full_scaled)

        return {
            "preds" : preds_all,
            "actual" : actual_all,
            "timeline" : timeline_ix,
            "final_preds" : final_preds,
            "scalers" : scalers
        }
    
    def plot_backtest(self, results):
        preds = results["preds"]
        actual = results["actual"]
        scalers = results["scalers"]

        preds_real = []
        actual_real = []
        for i in range(len(preds)):
            scaler = scalers[i]

            if i < 3:
                print(f"\n--- Step {i} ---")
                print("Scaled preds:", preds[i][:5])
                print("Actual (raw):", actual[i][:5])

            pred_inv = scaler.inverse_target(preds[i])
            actual_inv = actual[i]

            if i < 3:
                print("After inverse preds:", pred_inv[:5])
                print("After inverse actual:", actual_inv[:5])

            preds_real.append(pred_inv)
            actual_real.append(actual_inv)
        print("After inverse preds:", pred_inv[:5])
        print("After inverse actual:", actual_inv[:5])

        preds_real = np.array(preds_real)[:, 0]
        actual_real = np.array(actual_real)[:, 0]

        plt.figure(figsize=(12, 6))

        plt.plot(actual_real, label = "Actual", linewidth=2)
        plt.plot(preds_real, label = "Predicted", linestyle= "--")

        plt.title("Backtesting: Predicted vs Actual")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig("backtest_plot.png", dpi=300, bbox_inches="tight")

        plt.show()
