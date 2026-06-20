from .STCMOT import DLASeg, TDRM1, TEBM1


class IAEM(TEBM1):
    """Identity-Aware Attention Embedding Module."""

    pass


class TFRM(TDRM1):
    """Trajectory-Guided Feature Refinement Module."""

    pass


class ITTrack(DLASeg):
    """ITTrack network with paper-aligned module names."""

    def __init__(
        self,
        base_name,
        heads,
        pretrained,
        down_ratio,
        final_kernel,
        last_level,
        head_conv,
        out_channel=0,
    ):
        super().__init__(
            base_name=base_name,
            heads=heads,
            pretrained=pretrained,
            down_ratio=down_ratio,
            final_kernel=final_kernel,
            last_level=last_level,
            head_conv=head_conv,
            out_channel=out_channel,
        )
        self.reid_cnn = IAEM(128)
        self.hm_cnn = TFRM()


def ittrack_model(num_layers, heads, head_conv=256, down_ratio=4):
    model = ITTrack(
        base_name="dla{}".format(num_layers),
        heads=heads,
        pretrained=True,
        down_ratio=down_ratio,
        final_kernel=1,
        last_level=5,
        head_conv=head_conv,
    )
    return model
