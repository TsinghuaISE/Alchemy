import torch
import torch.nn as nn


class Model(nn.Module):
    """
    TimeEmb: Temporal embedding based forecaster adapted to time_series.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # Optional flags with defaults
        self.use_revin = getattr(configs, 'use_revin', 1)
        self.use_day_index = getattr(configs, 'use_day_index', 0)
        self.use_hour_index = getattr(configs, 'use_hour_index', 1)
        self.emb_len_hour = getattr(configs, 'hour_length', 24)
        self.emb_len_day = getattr(configs, 'day_length', 7)

        self.scale = 0.02

        layers = [
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        ]
        self.model = nn.Sequential(*layers)

        self.emb_hour = nn.Parameter(
            torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1),
            requires_grad=True
        )
        self.emb_day = nn.Parameter(
            torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1),
            requires_grad=True
        )
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

    def _build_indices(self, x_enc, x_mark_enc):
        """
        Extract hour/day index from time marks.
        - If timeenc=1 (normalized time features), denormalize to integer hour/day.
        - Else fallback to raw integer marks.
        """
        batch_size = x_enc.shape[0]
        device = x_enc.device

        # defaults
        hour_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        day_index = torch.zeros(batch_size, dtype=torch.long, device=device)

        if x_mark_enc is None:
            return hour_index, day_index

        # take the last time step's markers
        marks = x_mark_enc[:, -1, :]

        # Heuristic: timeenc=1 produces normalized features in [-0.5, 0.5]
        if torch.all(marks <= 1.1) and torch.all(marks >= -1.1) and marks.shape[-1] >= 2:
            # Expected order for hourly freq: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
            hour_norm = marks[:, 0]
            hour_index = torch.clamp(torch.round((hour_norm + 0.5) * 23), 0, 23).long().to(device)

            # DayOfWeek exists at index 1 for hourly freq; clamp to [0,6]
            day_norm = marks[:, 1]
            day_index = torch.clamp(torch.round((day_norm + 0.5) * 6), 0, 6).long().to(device)
        else:
            # Fallback: use raw integer marks if provided (legacy path)
            if marks.shape[-1] >= 4:
                hour_index = marks[:, 3].long().to(device)
                day_index = marks[:, 2].long().to(device)
            elif marks.shape[-1] >= 2:
                hour_index = marks[:, 1].long().to(device)
                day_index = marks[:, 0].long().to(device)

        return hour_index, day_index

    def forecast(self, x_enc, x_mark_enc):
        """
        x_enc: (B, L, D), x_mark_enc: (B, L, mark_dim) or None.
        """
        hour_index, day_index = self._build_indices(x_enc, x_mark_enc)
        x = x_enc

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(self.w, dim=1, norm='ortho')
        x_freq_real = x.real
        x_freq_imag = x.imag

        if self.use_hour_index:
            emb_hour = self.emb_hour[hour_index % self.emb_len_hour]
            x_freq_real = x_freq_real - emb_hour
        else:
            emb_hour = None

        if self.use_day_index:
            emb_day = self.emb_day[day_index % self.emb_len_day]
            x_freq_real = x_freq_real - emb_day
        else:
            emb_day = None

        x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
        y = x_freq_minus_emb * w
        y_real = y.real
        y_freq_imag = y.imag

        if self.use_day_index and emb_day is not None:
            y_real = y_real + emb_day
        if self.use_hour_index and emb_hour is not None:
            y_real = y_real + emb_hour

        y_freq = torch.complex(y_real, y_freq_imag)
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y = self.model(y).permute(0, 2, 1)

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        return None

