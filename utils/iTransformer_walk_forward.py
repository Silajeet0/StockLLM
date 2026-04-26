import copy
import numpy as np
import torch


class WalkForwardTrainer:
    def __init__(
        self,
        agent,                 # your existing iTransformerAgent
        data_array,            # scaled numpy array
        cutoff_idx,            # index for Oct 2022
        device,
        alpha=0.1,
    ):
        self.agent = agent
        self.data = data_array
        self.cutoff = cutoff_idx
        self.device = device
        self.alpha = alpha

        self.W_ema = None
        self.backtest_preds = []
        self.backtest_targets = []

        def _ema_update(self, old_w, new_w):
            if old_w is None:
                return copy.deepcopy(new_w)

            updated = {}
            for k in old_w:
                updated[k] = self.alpha * new_w[k] + (1 - self.alpha) * old_w[k]

            return updated
        
        def train_base(self):
            print("[WalkForward] Training base model...")
            model = self.agent._init_model()
            train_data = self.data[:self.cutoff]
            self.agent._train_model(model, train_data)
            base_weights = copy.deepcopy(model.state_dict())
            return base_weights
        
        def _predict_step(self, model, data_slice):
            x = torch.from_numpy(data_slice[-self.agent.seq_len:]) \
                    .unsqueeze(0).to(self.device).float()

            x_mark  = torch.zeros(1, self.agent.seq_len, 4, device=self.device)
            y_mark  = torch.zeros(1, self.agent.pred_len, 4, device=self.device)
            dec_inp = torch.zeros(
                1, self.agent.pred_len, self.agent.model.enc_in,
                device=self.device
            )

            model.eval()
            with torch.no_grad():
                out = model(x, x_mark, dec_inp, y_mark)

                if isinstance(out, (tuple, list)):
                    out = out[0]

                preds = out[:, -self.agent.pred_len:, 0].cpu().numpy()[0]

            return preds
        
        def walk_forward(self, base_weights, end_idx):
            print("[WalkForward] Starting walk-forward...")

            model = self.agent._init_model()
            model.load_state_dict(base_weights)

            self.W_ema = copy.deepcopy(base_weights)

            for d in range(self.cutoff, end_idx):

                train_data = self.data[:d]

                # warm start training
                self.agent._train_model(model, train_data)

                # prediction
                preds = self._predict_step(model, train_data)

                # actual
                actual = self.data[d:d + self.agent.pred_len, 0]

                self.backtest_preds.append(preds)
                self.backtest_targets.append(actual)

                # EMA update
                self.W_ema = self._ema_update(
                    self.W_ema,
                    model.state_dict()
                )

                print(f"[WalkForward] Step {d} done")

            return self.W_ema
        
        def train_final(self):
            print("[WalkForward] Training final model...")

            model = self.agent._init_model()
            model.load_state_dict(self.W_ema)

            self.agent._train_model(model, self.data)

            return model
        
        def final_predict(self, model):
            print("[WalkForward] Final prediction...")
            preds = self._predict_step(model, self.data)
            return preds
        
        def run(self, end_idx):
            base_w = self.train_base()

            self.walk_forward(base_w, end_idx)

            final_model = self.train_final()

            final_preds = self.final_predict(final_model)

            return {
                "final_predictions": final_preds,
                "backtest_preds": np.array(self.backtest_preds),
                "backtest_targets": np.array(self.backtest_targets),
            }