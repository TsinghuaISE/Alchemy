# -*- coding: utf-8 -*-

# ===== Inlined Components =====
class Decomposition(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5

        self.dwt = DWT1DForward(wave=self.wavelet_name, J=self.level,
                                use_amp=self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(
            wave=self.wavelet_name, J=self.level, use_amp=self.use_amp)
        self.idwt = DWT1DInverse(wave=self.wavelet_name,
                                 use_amp=self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(
            wave=self.wavelet_name, use_amp=self.use_amp)

        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [
            self.input_length]  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [
            self.pred_length]  # required length of the pred seq after decom

        self.tfactor = tfactor
        self.dfactor = dfactor
        #################################
        self.affine = False
        #################################

        if self.affine:
            self._init_params()

    def transform(self, x):
        # input: x shape: batch, channel, seq
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, []  # no decompose: returning the same value in yl
        return yl, yh

    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl  # no decompose: returning the same value in x
        return x

    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))

    def _wavelet_decompose(self, x):
        # input: x shape: batch, channel, seq
        yl, yh = self.dwt(x)

        if self.affine:
            yl = yl.transpose(1, 2)  # batch, seq, channel
            yl = yl * self.affine_weight[0]
            yl = yl + self.affine_bias[0]
            yl = yl.transpose(1, 2)  # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ * self.affine_weight[i + 1]
                yh_ = yh_ + self.affine_bias[i + 1]
                yh[i] = yh_.transpose(1, 2)  # batch, channel, seq

        return yl, yh

    def _wavelet_reverse_decompose(self, yl, yh):
        if self.affine:
            yl = yl.transpose(1, 2)  # batch, seq, channel
            yl = yl - self.affine_bias[0]
            yl = yl / (self.affine_weight[0] + self.eps)
            yl = yl.transpose(1, 2)  # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ - self.affine_bias[i + 1]
                yh_ = yh_ / (self.affine_weight[i + 1] + self.eps)
                yh[i] = yh_.transpose(1, 2)  # batch, channel, seq

        x = self.idwt((yl, yh))
        return x  # shape: batch, channel, seq

class DWT1DForward(nn.Module):
    """ Performs a 1d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """

    def __init__(self, J=1, wave='db1', mode='zero', use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]

        # Prepare the filters - this makes them into column filters
        filts = prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])
        self.register_buffer('h1', filts[1])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            x0, x1 = AFB1D.apply(x0, self.h0, self.h1, mode, self.use_amp)
            highs.append(x1)

        return x0, highs

class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave='db1', mode='zero', use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # Prepare the filters
        filts = prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = mode_to_int(self.mode)
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = SFB1D.apply(x0, x1, self.g0, self.g1, mode, self.use_amp)
        return x0

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n + end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:, -n:], x[:, :-n + end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:, :, -n:], x[:, :, :-n + end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:, :, :, -n:], x[:, :, :, :-n + end]), dim=3)

def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, :, xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1 + m2, dtype='int32'), -0.5, l1 - 0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2 + m2, dtype='int32'), -0.5, l2 - 0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:, :, xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:, :, :, xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def afb1d(x, h0, h1, use_amp, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            else:
                x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        if use_amp:
            with torch.cuda.amp.autocast():  # for mixed precision
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        else:
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:, :, :L2] = lohi[:, :, :L2] + lohi[:, :, N2:N2 + L2]
            lohi = lohi[:, :, :N2]
        else:
            lohi[:, :, :, :L2] = lohi[:, :, :, :L2] + lohi[:, :, :, N2:N2 + L2]
            lohi = lohi[:, :, :, :N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            # Calculate the high and lowpass
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi

def sfb1d(lo, hi, g0, g1, use_amp, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)
    if mode == 'per' or mode == 'periodization':
        if use_amp:
            with torch.cuda.amp.autocast():
                y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, groups=C)
        else:
            y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:, :, :L - 2] = y[:, :, :L - 2] + y[:, :, N:N + L - 2]
            y = y[:, :, :N]
        else:
            y[:, :, :, :L - 2] = y[:, :, :, :L - 2] + y[:, :, :, N:N + L - 2]
            y = y[:, :, :, :N]
        y = roll(y, 1 - L // 2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            if use_amp:
                with torch.cuda.amp.autocast():
                    y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                        F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
            else:
                y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y

def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

class AFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """

    @staticmethod
    def forward(ctx, x, h0, h1, mode, use_amp):
        mode = int_to_mode(mode)

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]
        ctx.mode = mode
        ctx.use_amp = use_amp

        lohi = afb1d(x, h0, h1, use_amp, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors
            use_amp = ctx.use_amp

            # Make grads 4d
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]

            dx = sfb1d(dx0, dx1, h0, h1, use_amp, mode=mode, dim=3)[:, :, 0]

            # Check for odd input
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, :ctx.shape]

        return dx, None, None, None, None, None

def afb2d(x, filts, mode='zero'):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2, 3)
            h1_col = h1
            h1_row = h1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
    y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)

    return y

def sfb2d(ll, lh, hl, hh, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.sfb1d`

    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2, 3)
            g1_col = g1
            g1_row = g1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y

class SFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        low (torch.Tensor): Lowpass to reconstruct of shape (N, C, L)
        high (torch.Tensor): Highpass to reconstruct of shape (N, C, L)
        g0: lowpass
        g1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*2, L')
    """

    @staticmethod
    def forward(ctx, low, high, g0, g1, mode, use_amp):
        mode = int_to_mode(mode)
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        ctx.mode = mode
        ctx.save_for_backward(g0, g1)
        ctx.use_amp = use_amp

        return sfb1d(low, high, g0, g1, use_amp, mode=mode, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            use_amp = ctx.use_amp
            g0, g1, = ctx.saved_tensors
            dy = dy[:, :, None, :]

            dx = afb1d(dy, g0, g1, use_amp, mode=mode, dim=3)

            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None

def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row

def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row

def prep_filt_afb1d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)
# ===== End Inlined Components =====

"""
Created on Sun Jan  5 16:10:01 2025
@author: Murad
SISLab, USF
mmurad@usf.edu
https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
"""

import torch.nn as nn
import pywt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import torch

class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x

class Mixer(nn.Module):
    def __init__(self,
                 input_seq=[],
                 out_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 tfactor=[],
                 dfactor=[]):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor  # expansion factor for patch mixer
        self.dfactor = dfactor  # expansion factor for embedding mixer

        self.tMixer = TokenMixer(input_seq=self.input_seq, batch_size=self.batch_size, channel=self.channel,
                                 pred_seq=self.pred_seq, dropout=self.dropout, factor=self.tfactor,
                                 d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: output: [Batch, Channel, Patch_number, d_model]

        '''
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x

class ResolutionBranch(nn.Module):
    def __init__(self,
                 input_seq=[],
                 pred_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 patch_len=[],
                 patch_stride=[]):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)  # shared among all channels
        self.mixer1 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.mixer2 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]

        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''

        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)

        out = self.head(out)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                                                pred_seq=self.pred_w_dim[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim))])

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT

class Model(nn.Module):
    def __init__(self, args, tfactor=5, dfactor=5, wavelet='db2', level=1, stride=8, no_decomposition=False):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.wpmixerCore = WPMixerCore(input_length=self.args.seq_len,
                                       pred_length=self.args.pred_len,
                                       wavelet_name=wavelet,
                                       level=level,
                                       batch_size=self.args.batch_size,
                                       channel=self.args.c_out,
                                       d_model=self.args.d_model,
                                       dropout=self.args.dropout,
                                       embedding_dropout=self.args.dropout,
                                       tfactor=tfactor,
                                       dfactor=dfactor,
                                       device=self.args.device,
                                       patch_len=self.args.patch_len,
                                       patch_stride=stride,
                                       no_decomposition=no_decomposition,
                                       use_amp=self.args.use_amp)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]

        # De-Normalization
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out  # [B, L, D]
