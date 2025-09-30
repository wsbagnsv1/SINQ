# modified by SINQ authors 2025

import copy
from typing import Union

import torch
from torch import nn, Tensor

from . quantizer import Quantizer
from .utils import is_divisible


class SINQLinear(nn.Module):
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: dict,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_unpack_kernel: bool = False,
        layer_activations  = None
    ):
        super().__init__()

        self.set_forward_backend("pytorch")

        self.bias = None
        self.axis = None
        self.channel_wise = None
        self.device = device
        self.compute_dtype = compute_dtype
        self.quant_config = copy.deepcopy(quant_config)
        self.del_orig = del_orig
        self.use_unpack_kernel = use_unpack_kernel

        self.linear_layer = linear_layer
        self.W_q = None
        self.meta = None
        self.layer_activations = layer_activations

        self.initialize()

    def initialize(self):

        # Handle group_size==None
        if self.quant_config["weight_quant_params"]["group_size"] == None:
            self.quant_config["weight_quant_params"]["group_size"] = (
                self.linear_layer.in_features
                if (self.quant_config["weight_quant_params"]["axis"] == 1)
                else self.linear_layer.out_features
            )

        self.quantize(self.linear_layer.weight.data, self.layer_activations, **self.quant_config)
        self.bias = (
            None
            if (self.linear_layer.bias is None)
            else self.linear_layer.bias.clone().to(
                device=self.device, dtype=self.compute_dtype
            )
        )

        # Clear-up parameters
        if self.del_orig:
            for name, param in self.linear_layer.named_parameters():
                setattr(self.linear_layer, name, None)
            del self.linear_layer
            torch.cuda.empty_cache()

    def quantize(
        self,
        W: Tensor,
        layer_activations,
        weight_quant_params: dict,
        scale_quant_params: dict,
        zero_quant_params: dict,
    ) -> None:
        quant_scale = scale_quant_params is not None
        quant_zero = zero_quant_params is not None

        self.in_features, self.out_features = W.t().shape


        # Quantize
        # DEBUG
        # print("before Quantizer.quantize, self.device", self.device)
        W_q, meta = Quantizer.quantize(
            W,
            layer_activations,
            device=self.device,
            compute_dtype=self.compute_dtype,
            **weight_quant_params,
            use_unpack_kernel = self.use_unpack_kernel        
        )

        meta.update({"quant_scale": quant_scale, "quant_zero": quant_zero})

        self.W_q = W_q
        self.meta = meta
        self.ready = True

    def unpack(self, reshape=False, dtype=None):
        if self.ready is False:
            return None
        if self.meta["packing"]:
            W_r = Quantizer.unpack[self.meta["packing"]](
                self.W_q, dtype=dtype if (dtype is not None) else self.compute_dtype
            )
            return W_r.view(self.meta["shape"]) if (reshape) else W_r

    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta
        device = W_q.device

        del_keys = set()

        # Zero/Scale packed together
        if "zero_scale" in meta:
            zero_scale = meta["zero_scale"].to(device=device)

            if zero_scale.dtype == torch.uint8:
                meta["zero_q"], meta["scale_q"] = zero_scale[0], zero_scale[1]
                del_keys.update({"zero_q", "scale_q"})
            else:
                meta["zero"], meta["scale"] = zero_scale[0], zero_scale[1]
                del_keys.update({"zero", "scale"})

        if meta["quant_zero"]:
            meta["zero"] = Quantizer.dequantize(
                meta["zero_q"].to(device=device), meta["meta_zero"]
            )
            del_keys.add("zero")

        if meta["quant_scale"]:
            meta["scale"] = Quantizer.dequantize(
                meta["scale_q"].to(device=device), meta["meta_scale"]
            )
            del_keys.add("scale")

        W_est = Quantizer.dequantize(W_q, meta, use_unpack_kernel=self.use_unpack_kernel)

        # DEBUG
        # print("W_q.device", W_est.device)

        # Cleanup
        for key in del_keys:
            del meta[key]
        return W_est

    def forward_pytorch(self, x: Tensor) -> Tensor:
        # DEBUG
        # print(x.device, self.dequantize().device)

        out = torch.matmul(x, self.dequantize().t())
        if self.bias is not None:
            out += self.bias
        return out

    def forward_ascendc(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def set_forward_backend(cls, backend: str):
        if backend == "pytorch":
            cls.forward = cls.forward_pytorch
        elif backend == "ascendc":
            cls.forward = cls.forward_ascendc

    def extra_repr(self) -> str:
        out = ""
        if hasattr(self, "meta"):
            if self.meta is not None:
                in_features, out_features = self.meta["shape"][::-1]
                out = (
                    f"in_features={in_features}, out_features={out_features}, bias={self.bias is not None}, "
                    f"device={self.device}, W_q.device={self.W_q.device}"
                )
        return out


def sinq_base_quant_config(
    nbits: int = 4,
    group_size: int = 64,
    quant_zero: bool = False,
    quant_scale: bool = False,
    offload_meta: bool = False,  # meta-data should be quantized with the same settings to use offload_meta
    view_as_float: bool = False,
    axis: int = 1,
    tiling_mode: str = '1D',
    method: str = 'dual',
):
    assert (
        nbits in Quantizer.SUPPORTED_BITS
    ), "nbits value not supported. Check Quantizer.SUPPORTED_BITS."
    if method == "asinq":
        #Re-mapping to let user use sinq_awq_l1_quantAux as asinq (A-SINQ in the paper)
        method = "sinq_awq_l1_quantAux"
    if group_size is not None:
        assert is_divisible(
            group_size, 8
        ), "Invalid group_size param: the value should be a multiple of 8."

    weight_quant_params = {
        "nbits": nbits,
        "channel_wise": True,
        "group_size": group_size,
        "optimize": True,
        "round_zero": True if nbits == 4 else False,
        "axis": axis,
        "view_as_float": view_as_float,
        "tiling_mode": tiling_mode,
        "method": method,
    }

    if quant_zero or quant_scale:
        print(
            colored(
                "Warning: Quantized meta-data is deprecated and will be removed. It is not supported for quantized model serialization.",
                "yellow",
            )
        )

    scale_quant_params = (
        {"nbits": 8, "channel_wise": True, "group_size": 128, "optimize": False}
        if (quant_scale)
        else None
    )
    zero_quant_params = (
        {"nbits": 8, "channel_wise": False, "group_size": None, "optimize": False}
        if (quant_zero)
        else None
    )

    return {
        "weight_quant_params": weight_quant_params,
        "scale_quant_params": scale_quant_params,
        "zero_quant_params": zero_quant_params
    }


# Alias: follow similar Auto-GPTQ naming
BaseQuantizeConfig = sinq_base_quant_config
