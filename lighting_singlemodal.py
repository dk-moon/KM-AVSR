import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
import torch.nn.functional as F
import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_
import numpy as np

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def calculate_levenshtein_distance(reference, hypothesis):
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    dp = np.zeros((ref_len + 1, hyp_len + 1), dtype=np.int)

    for i in range(1, ref_len + 1):
        dp[i][0] = i
    for j in range(1, hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[ref_len][hyp_len]

def cer(reference, hypothesis):
    edit_distance = calculate_levenshtein_distance(reference, hypothesis)
    return edit_distance / len(reference)


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        self.sentence_count = 0
        self.total_cer = 0

        self.fout = open('result.txt','w')
        
        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)
        #enc_cp = enc_feat.detach().cpu().numpy()
        #np.save('enc_feat.npy',enc_cp) 

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]
        # actual = self.text_transform.post_process(token_id)
        actual = self.text_transform.post_process(token_id).replace("<eos>", "").replace("<unk>", "")

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        if self.cfg.label_flag==1:
            print(f'actual   :{actual}')
        print(f'predicted:{predicted}')
        self.fout.write(f'predicted:{predicted}\n')
        self.sentence_count+=1
        sentence_cer = cer(actual, predicted)
        print(f'CER: {sentence_cer}\n')
        self.total_cer += sentence_cer
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        with torch.no_grad():
            for i in range(batch_size):
                enc_feat, _ = self.model.encoder(batch["inputs"][i].unsqueeze(0).to(self.device), None)
                enc_feat = enc_feat.squeeze(0)
                nbest_hyps = self.beam_search(enc_feat)
                nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
                predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
                predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "").replace("<unk>", "")
                target = self.text_transform.post_process(batch["targets"][i]).replace("<eos>", "").replace("<unk>", "")
                sentence_cer = cer(target, predicted)
                if step_type == "train":
                    if not hasattr(self, "train_total_cer"): self.train_total_cer = 0.0
                    if not hasattr(self, "train_sentence_count"): self.train_sentence_count = 0
                    self.train_total_cer += sentence_cer
                    self.train_sentence_count += 1
                elif step_type == "val":
                    if not hasattr(self, "val_total_cer"): self.val_total_cer = 0.0
                    if not hasattr(self, "val_sentence_count"): self.val_sentence_count = 0
                    self.val_total_cer += sentence_cer
                    self.val_sentence_count += 1

        return loss

    def on_train_epoch_start(self):
        train_dataloader = self.trainer.train_dataloader

        # 일부 상황에서는 CombinedLoader가 사용되므로 그 경우를 분기 처리
        if hasattr(train_dataloader, "loaders"):
            # CombinedLoader: train_dataloader.loaders is a dict or list
            for loader in train_dataloader.loaders.values() if isinstance(train_dataloader.loaders, dict) else train_dataloader.loaders:
                sampler = getattr(loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.current_epoch)
        else:
            # 일반 DataLoader
            sampler = getattr(train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        if hasattr(self, "train_sentence_count") and self.train_sentence_count > 0:
            avg_cer = self.train_total_cer / self.train_sentence_count
            self.log("cer_train_epoch", avg_cer)
            self.train_total_cer = 0.0
            self.train_sentence_count = 0
            print(f"[Train Epoch {self.current_epoch}] CER: {avg_cer:.4f}")

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_validation_epoch_end(self):
        if hasattr(self, "val_sentence_count") and self.val_sentence_count > 0:
            avg_cer = self.val_total_cer / self.val_sentence_count
            self.log("cer_val_epoch", avg_cer)
            self.val_total_cer = 0.0
            self.val_sentence_count = 0
            print(f"[Val Epoch {self.current_epoch}] CER: {avg_cer:.4f}")

    def on_test_epoch_end(self):
        if self.cfg.label_flag==1:
            self.log("wer", self.total_edit_distance / self.total_length)
            self.log("cer", self.total_cer / float(self.sentence_count))
        else:
            self.log("sentence_count", self.sentence_count)
            self.fout.close()



def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
