import torch
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.nets_utils import MLPHead

class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        super().__init__()

        self.encoder = Encoder(
            attention_dim=args["adim"],
            attention_heads=args["aheads"],
            linear_units=args["eunits"],
            num_blocks=args["elayers"],
            input_layer=args["transformer_input_layer"],
            dropout_rate=args["dropout_rate"],
            positional_dropout_rate=args["dropout_rate"],
            attention_dropout_rate=args["transformer_attn_dropout_rate"],
            encoder_attn_layer_type=args["transformer_encoder_attn_layer_type"],
            macaron_style=args["macaron_style"],
            use_cnn_module=args["use_cnn_module"],
            cnn_module_kernel=args["cnn_module_kernel"],
            zero_triu=args.get("zero_triu", False),
            a_upsample_ratio=args["a_upsample_ratio"],
            relu_type=args.get("relu_type", "swish"),
        )

        self.aux_encoder = Encoder(
            attention_dim=args.get("aux_adim", args["adim"]),
            attention_heads=args.get("aux_aheads", args["aheads"]),
            linear_units=args.get("aux_eunits", args["eunits"]),
            num_blocks=args.get("aux_elayers", args["elayers"]),
            input_layer=args.get("aux_transformer_input_layer", args["transformer_input_layer"]),
            dropout_rate=args.get("aux_dropout_rate", args["dropout_rate"]),
            positional_dropout_rate=args.get("aux_dropout_rate", args["dropout_rate"]),
            attention_dropout_rate=args.get("aux_transformer_attn_dropout_rate", args["transformer_attn_dropout_rate"]),
            encoder_attn_layer_type=args.get("aux_transformer_encoder_attn_layer_type", args["transformer_encoder_attn_layer_type"]),
            macaron_style=args.get("aux_macaron_style", args["macaron_style"]),
            use_cnn_module=args.get("aux_use_cnn_module", args["use_cnn_module"]),
            cnn_module_kernel=args.get("aux_cnn_module_kernel", args["cnn_module_kernel"]),
            zero_triu=args.get("aux_zero_triu", args.get("zero_triu", False)),
            a_upsample_ratio=args.get("aux_a_upsample_ratio", args["a_upsample_ratio"]),
            relu_type=args.get("aux_relu_type", args.get("relu_type", "swish")),
        )

        self.fusion = MLPHead(
            idim=args["adim"] * 2,
            hdim=args["adim"],
            odim=args["adim"],
            norm=None,
        )

        self.proj_decoder = None
        if args["adim"] != args["ddim"]:
            self.proj_decoder = torch.nn.Linear(args["adim"], args["ddim"])

        self.decoder = Decoder(
            odim=odim,
            attention_dim=args["ddim"],
            attention_heads=args["dheads"],
            linear_units=args["dunits"],
            num_blocks=args["dlayers"],
            dropout_rate=args["dropout_rate"],
            positional_dropout_rate=args["dropout_rate"],
            self_attention_dropout_rate=args["transformer_attn_dropout_rate"],
            src_attention_dropout_rate=args["transformer_attn_dropout_rate"],
        )

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        self.criterion = LabelSmoothingLoss(
            odim, ignore_id, args["lsm_weight"], args["transformer_length_normalized_loss"]
        )
        self.ctc = CTC(
            odim, args["adim"], args["dropout_rate"], ctc_type=args["ctc_type"], reduce=True
        )
        self.mtlalpha = args["mtlalpha"]

    def forward(self, video, audio, video_lengths, audio_lengths, label):
        video_mask = make_non_pad_mask(video_lengths).to(video.device).unsqueeze(-2)
        video_feat, _ = self.encoder(video, video_mask)

        audio_lengths = torch.div(audio_lengths, 640, rounding_mode="trunc")
        audio_mask = make_non_pad_mask(audio_lengths).to(audio.device).unsqueeze(-2)
        audio_feat, _ = self.aux_encoder(audio, audio_mask)

        x = self.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        loss_ctc, _ = self.ctc(x, video_lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, video_mask)

        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)
        return loss, loss_ctc, loss_att, acc