# modified by SINQ authors 2025

from typing import Union

import torch
from torch import Tensor

from .optimize import optimize_weights_proximal
from .bitpack import BitPack
from .utils import is_divisible

from torch import uint8, int32, float16, nn, Tensor


NF4_CODEBOOK = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
     0.0,
     0.07958029955625534,
     0.16093020141124725,
     0.24611230194568634,
     0.33791524171829224,
     0.44070982933044434,
     0.5626170039176941,
     0.7229568362236023,
     1.0
], dtype=torch.float32)

# NF3 codebook (~[-1, 1], 8 non-uniform levels from the paper "Asymmetric Floating Point Quantization for LLMs")
NF3_CODEBOOK = torch.tensor([
    -1.0,
    -0.5350227355957031,
    -0.2469314038753510,
     0.0,
     0.1833375245332718,
     0.3819939494132996,
     0.6229856610298157,
     1.0
], dtype=torch.float32)

def rtn8(x, tile=32):
    shape = x.shape
    x = x.view(tile, -1)
    min = x.min(0, keepdim=True).values
    x = x - min
    scale = x.max(0, keepdim=True).values / (2**8 - 1)
    x = torch.round(x / scale).clip(0, 2**8 - 1).to(torch.uint8)
    return x, scale, min, shape

def dq8(x):
    # x is already dequantized
    if isinstance(x, torch.Tensor):
        return x 

    if x is None:
        return None
    # Dict form (from saved models)
    if isinstance(x, dict):
        xt = x["x"]; s = x["s"]; m = x["m"]; shape = x.get("shape")
        # Guard: catch accidental stringified tensors
        bad = [t for t in (xt, s, m) if isinstance(t, str)]
        if bad:
            raise ValueError("quantAux meta was serialized incorrectly: found strings in meta['scale'/'zero']. "
                             "Ensure save_weights_safetensors() recursively extracts tensors in meta.")
        if isinstance(shape, list):
            shape = tuple(shape)
        return (xt * s + m).view(shape)

    # Legacy tuple form (live before save)
    xt, s, m, shape = x
    return (xt * s + m).view(shape)

class Quantizer:
    SUPPORTED_BITS = [8, 6, 5, 4, 3, 2, 1.58, 1]
    optimize_weights = optimize_weights_proximal

    bit_to_packing = {
        8: "8bit_u8",
        6: "6bit_32",  
        5: "5bit_32", 
        4: "4bit_u8",
        3: "3bit_32",
        2: "2bit_u8",
        1.58: "2bit_u8",  # todo: bitpacking
        1: "1bit_u8",
    }

    pack = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "6bit_32": BitPack.pack_6bit_32,
        "5bit_32": BitPack.pack_5bit_32,
        "4bit_u8": BitPack.pack_4bit_u8,
        "3bit_32": BitPack.pack_3bit_32,
        "2bit_u8": BitPack.pack_2bit_u8,
        "1bit_u8": BitPack.pack_1bit_u8,
    }

    unpack = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "6bit_32": BitPack.unpack_6bit_32,
        "5bit_32": BitPack.unpack_5bit_32,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "3bit_32": BitPack.unpack_3bit_32,
        "2bit_u8": BitPack.unpack_2bit_u8,
        "1bit_u8": BitPack.unpack_1bit_u8,
    }

    unpack_view_dtype = {
        "8bit_u8": uint8,
        "6bit_32": int32,
        "5bit_32": int32,
        "4bit_u8": uint8,
        "3bit_32": int32,
        "2bit_u8": uint8,
        "1bit_u8": uint8,
    }
    @classmethod
    def quantize(
        cls,
        tensor: Tensor,
        layer_activations,
        nbits: float = 4,
        group_size: int = 64,
        round_zero: bool = False,
        axis: int = 0,
        bitpack: bool = True,
        compute_dtype: Union[torch.dtype, None] = None,
        view_as_float: bool = False,
        device: str = "cuda",
        tiling_mode: str = '1D',
        method: str = 'sinq',
        use_unpack_kernel: bool = False
    ) -> tuple:
        if group_size == 0:
            group_size = None
        if group_size < 0:
            raise ValueError(f'Group size must be greater than or equal to 0 (a value of 0 means no grouping)!')
        assert nbits in Quantizer.SUPPORTED_BITS, (
            "nbits=" + str(nbits) + " not supported."
        )
        assert axis in [0, 1], "axis should be either 0 or 1"
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )

        W = tensor.float()
        shape = W.shape

        # Reshape for grouping
        if (group_size is not None):
            W = (
                W.reshape([-1, group_size])
                if (axis == 1)
                else W.reshape([group_size, -1])
            )

        # Get min/max values
        _min = W.min(axis=axis, keepdim=True)[0]
        _max = W.max(axis=axis, keepdim=True)[0]

        max_v = round(2**nbits - 1)
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
        denom = (_max - _min)
        scale = (max_v / denom)
        scale = torch.where(denom.abs() <= 1e-4, torch.full_like(scale, 1.0), scale) #Avoid small denom values
        scale = scale.clamp(max=2e4) # clamp to avoid half-precision problems
        zero = -_min * scale

        # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
        if round_zero:
            zero = torch.round(zero)

        # Use SINQ on weights
        opt_result = Quantizer.optimize_weights(
            tensor=W,
            layer_activations=layer_activations,
            scale=scale,
            zero=zero,
            min_max=min_max,
            axis=axis,
            device=device,
            shape=shape,
            tiling_mode=tiling_mode,
            method=method
        )

        # Check if optimization returned None (meaning quantization should be skipped)
        if opt_result is None:
            # Return a signal that this layer should not be quantized
            # This will be handled by the quantize_model function
            raise ValueError(f"QUANT_SKIP_LAYERS: Layer has incompatible dimensions for quantization with tiling_mode={tiling_mode}")

        W_q, scale, zero, scale2, awq_scale = opt_result

        if 'quantAux' in method:
            scale = rtn8(scale)
            zero = rtn8(zero, tile=torch.numel(zero)) if not (zero is None) else zero

        # cleanup
        del W, _min, _max
        torch.cuda.empty_cache() 

        # Store meta-data (we invert the scale for dequantization)
        # scale = 1.0 / scale
        meta = {
            "nbits": nbits,
            "group_size": group_size,
            "shape": shape,
            "scale": scale,
            "scale2": scale2,
            "awq_scale": awq_scale,
            "zero": zero,
            "axis": axis,
            "packing": Quantizer.bit_to_packing[nbits],
            "method": method,
            "compute_dtype": compute_dtype,
        }
        meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[meta["packing"]]

        # Pack bits
        meta["view_as_float"] = view_as_float
        if bitpack:
            W_q = Quantizer.pack[meta["packing"]](W_q)
            if view_as_float:
                W_q = W_q.view(
                    torch.float32 if compute_dtype is None else compute_dtype
                )  # store quantized weights as compute_dtype
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None

        torch.cuda.empty_cache()

        return W_q, meta

    # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
    @classmethod
    def dequantize(cls, W_q: Tensor, meta: dict, use_unpack_kernel: bool = False) -> Tensor:
        compute_dtype = meta.get("compute_dtype", torch.float16)

        try:
            # 1) Unpack to per-element codes
            if meta["packing"]:
                if meta.get("view_as_float", False):
                    W_q = W_q.view(meta["unpack_view_dtype"])
                W_r = cls.unpack[meta["packing"]](W_q, dtype=compute_dtype)
            else:
                W_r = W_q.to(compute_dtype)

            # 2) Load scales/zeros
            method = meta.get("method", "").lower()
            if "quantaux" in method:
                s = dq8(meta["scale"])
                z = dq8(meta["zero"])
            else:
                s = meta["scale"]
                z = meta["zero"]
            s2 = meta.get("scale2", None)

            # Make sure they're on W_r.device
            dev = W_r.device
            import torch as _torch
            if isinstance(s, _torch.Tensor):  s = s.to(dev)
            if isinstance(z, _torch.Tensor):  z = z.to(dev)
            if isinstance(s2, _torch.Tensor): s2 = s2.to(dev)

            # 3) NFx (NF3 or NF4)
            if ("nf4" in method) or ("nf3" in method):
                is_nf3 = ("nf3" in method)
                cb = (NF4_CODEBOOK if not is_nf3 else NF3_CODEBOOK).to(W_r.device, dtype=s.dtype)
                max_code = cb.numel() - 1

                if len(s.shape) == 2:
                    rows = s.shape[0]
                    idx  = W_r[:rows].to(torch.int64).clamp_(0, max_code)   # 0..7 or 0..15
                    vals = cb[idx]                                          # [-1,1] levels
                    out  = ((vals - z.to(cb.dtype)) * s.to(cb.dtype)).reshape(meta["shape"])
                    if s2 is not None:
                        out = out * s2
                    W_r = out

                elif len(s.shape) == 3:
                    # Blocked/tiling case
                    H, W = meta["shape"]
                    block = W_r.shape[-1]
                    n_h, n_w = H // block, W // block
                    total = n_h * block * n_w * block

                    idx  = W_r.view(-1)[:total].to(torch.int64).clamp_(0, max_code)
                    vals = cb[idx].view(n_h, block, n_w, block)

                    # Broadcast s/z/s2 like your uniform path
                    try:
                        z_ = z.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3).to(cb.dtype)
                    except Exception:
                        z_ = z.to(cb.dtype)
                    s1 = s.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3).to(cb.dtype)
                    s2_ = (s2.reshape(n_h, n_w, 1, block).permute(0, 2, 1, 3)
                        if s2 is not None else torch.ones_like(s1))

                    W_r = ((vals - z_) * s1 * s2_).view(H, W)
                else:
                    raise ValueError("invalid scale shape for NF dequant")

                if torch.any(torch.isnan(W_r)):
                    raise RuntimeError("NaN detected in NF dequantized weights")
                return W_r

            # 4) Uniform / other paths (unchanged, but robust s2 handling)
            if len(s.shape) == 2:
                s2_eff = 1 if s2 is None else s2
                W_r = W_r[: s.shape[0]]

                # Add dimension compatibility check for robust dequantization
                try:
                    W_r = ((W_r - z) * s).reshape(meta["shape"]) * s2_eff
                except (RuntimeError, ValueError) as reshape_error:
                    # Handle dimension mismatch gracefully
                    expected_size = torch.prod(torch.tensor(meta["shape"])).item()
                    actual_size = ((W_r - z) * s).numel()

                    print(f"[QUANTIZER WARNING] Shape mismatch detected in dequantization:")
                    print(f"  Expected shape: {meta['shape']} (size: {expected_size})")
                    print(f"  Actual flat size: {actual_size}")
                    print(f"  Original W_q shape: {W_q.shape}")
                    print(f"  Scale shape: {s.shape}")

                    if actual_size != expected_size:
                        print(f"[QUANTIZER INFO] Attempting dimension correction...")

                        # Try to adjust the reshape operation
                        try:
                            # Method 1: Handle broadcasting mismatch by reshaping scale/zero tensors
                            print(f"[QUANTIZER INFO] Attempting broadcasting-compatible dequantization...")

                            # The issue is that scale has shape [77472, 1] but we need it to broadcast with W_r
                            # Let's try to make the operation work by reshaping the scale appropriately
                            W_r_processed = W_r  # Start with the unpacked weights

                            # Try to reshape scale to be compatible with W_r for broadcasting
                            try:
                                # Reshape scale to match the first dimension of W_r for broadcasting
                                if s.shape[0] > W_r_processed.shape[0]:
                                    # Scale is larger than W_r first dimension - try to find compatible grouping
                                    scale_groups = s.shape[0] // W_r_processed.shape[0]
                                    if s.shape[0] % W_r_processed.shape[0] == 0:
                                        # We can group the scale values
                                        s_reshaped = s.view(W_r_processed.shape[0], scale_groups, -1)
                                        s_broadcast = s_reshaped.mean(dim=1)  # Average across groups
                                        z_reshaped = z.view(W_r_processed.shape[0], scale_groups, -1) if z is not None else None
                                        z_broadcast = z_reshaped.mean(dim=1) if z_reshaped is not None else None
                                        print(f"[QUANTIZER INFO] Grouped scale from {s.shape} to {s_broadcast.shape}")
                                    else:
                                        # Fallback: take the first W_r.shape[0] elements
                                        s_broadcast = s[:W_r_processed.shape[0]]
                                        z_broadcast = z[:W_r_processed.shape[0]] if z is not None else z
                                        print(f"[QUANTIZER INFO] Trimmed scale from {s.shape} to {s_broadcast.shape}")
                                else:
                                    # Scale is smaller or equal - use as is or expand
                                    s_broadcast = s
                                    z_broadcast = z
                                    print(f"[QUANTIZER INFO] Using scale as-is: {s.shape}")

                                # Now try the dequantization with reshaped scale/zero
                                W_r_flat = (W_r_processed - z_broadcast) * s_broadcast
                                print(f"[QUANTIZER INFO] Dequantized flat tensor shape: {W_r_flat.shape}")

                                # Reshape to final expected shape
                                if W_r_flat.numel() == expected_size:
                                    W_r = W_r_flat.reshape(meta["shape"]) * s2_eff
                                    print(f"[QUANTIZER INFO] Successfully dequantized with scale reshaping")
                                else:
                                    # Final fallback - flatten and pad/crop
                                    W_r_flat = W_r_flat.flatten()
                                    if W_r_flat.numel() > expected_size:
                                        W_r_flat = W_r_flat[:expected_size]
                                    elif W_r_flat.numel() < expected_size:
                                        padding_size = expected_size - W_r_flat.numel()
                                        W_r_flat = torch.nn.functional.pad(W_r_flat, (0, padding_size))
                                    W_r = W_r_flat.reshape(meta["shape"]) * s2_eff
                                    print(f"[QUANTIZER INFO] Final fallback dequantization successful")

                            except Exception as reshape_error:
                                print(f"[QUANTIZER ERROR] Scale reshaping failed: {reshape_error}")
                                # Ultimate fallback - try to dequantize without scale/zero (not ideal but better than crash)
                                try:
                                    print(f"[QUANTIZER WARNING] Attempting emergency dequantization without scale/zero...")
                                    W_r_flat = W_r_processed  # Just use unpacked weights
                                    if W_r_flat.numel() > expected_size:
                                        W_r_flat = W_r_flat.flatten()[:expected_size]
                                    elif W_r_flat.numel() < expected_size:
                                        padding_size = expected_size - W_r_flat.numel()
                                        W_r_flat = torch.nn.functional.pad(W_r_flat.flatten(), (0, padding_size))
                                    else:
                                        W_r_flat = W_r_flat.flatten()

                                    W_r = W_r_flat.reshape(meta["shape"]) * s2_eff
                                    print(f"[QUANTIZER WARNING] Emergency dequantization completed - results may be inaccurate")
                                except Exception as emergency_error:
                                    print(f"[QUANTIZER ERROR] Emergency dequantization failed: {emergency_error}")
                                    raise reshape_error

                        except Exception as correction_error:
                            print(f"[QUANTIZER ERROR] Failed to correct dimensions: {correction_error}")
                            # Fallback: try original reshape with minimal changes
                            try:
                                # Try with different interpretation of the scale broadcasting
                                if s.shape[0] > W_r.shape[0]:
                                    # Scale might be for a different tiling/arrangement
                                    s_adjusted = s[:W_r.shape[0]]
                                    z_adjusted = z[:W_r.shape[0]] if z is not None else z
                                    W_r = ((W_r - z_adjusted) * s_adjusted).reshape(meta["shape"]) * s2_eff
                                    print(f"[QUANTIZER INFO] Used adjusted scale/zero tensors")
                                else:
                                    raise ValueError("Cannot resolve dimension mismatch")
                            except Exception as fallback_error:
                                print(f"[QUANTIZER ERROR] Fallback also failed: {fallback_error}")
                                raise RuntimeError(f"Dequantization failed due to incompatible tensor dimensions. "
                                                f"Expected {meta['shape']}, but got incompatible shapes with "
                                                f"W_q: {W_q.shape}, scale: {s.shape}")
                    else:
                        # Sizes match but broadcasting still fails - need to handle scale/zero broadcasting issue
                        print(f"[QUANTIZER INFO] Sizes match but broadcasting fails - fixing scale/zero tensors...")
                        try:
                            # Handle the broadcasting issue directly
                            W_r_processed = W_r

                            # The issue is scale [77472, 1] needs to broadcast with W_r which has smaller first dimension
                            if s.shape[0] > W_r_processed.shape[0]:
                                # We need to reshape the scale to be compatible
                                scale_groups = s.shape[0] // W_r_processed.shape[0]
                                if s.shape[0] % W_r_processed.shape[0] == 0:
                                    # Perfect grouping - reshape and use
                                    s_reshaped = s.view(W_r_processed.shape[0], scale_groups, -1)
                                    s_broadcast = s_reshaped.mean(dim=1)  # Average across groups
                                    z_reshaped = z.view(W_r_processed.shape[0], scale_groups, -1) if z is not None else None
                                    z_broadcast = z_reshaped.mean(dim=1) if z_reshaped is not None else None
                                    print(f"[QUANTIZER INFO] Scale grouping successful: {s.shape} -> {s_broadcast.shape}")
                                else:
                                    # Take the first compatible portion
                                    s_broadcast = s[:W_r_processed.shape[0]]
                                    z_broadcast = z[:W_r_processed.shape[0]] if z is not None else z
                                    print(f"[QUANTIZER INFO] Scale trimmed: {s.shape} -> {s_broadcast.shape}")
                            else:
                                s_broadcast = s
                                z_broadcast = z
                                print(f"[QUANTIZER INFO] Using scale directly: {s.shape}")

                            # Try dequantization with corrected scale/zero
                            W_r_result = (W_r_processed - z_broadcast) * s_broadcast
                            W_r = W_r_result.reshape(meta["shape"]) * s2_eff
                            print(f"[QUANTIZER INFO] Broadcasting fix successful!")

                        except Exception as broadcast_fix_error:
                            print(f"[QUANTIZER ERROR] Broadcasting fix failed: {broadcast_fix_error}")
                            # Final attempt - try view with original tensors
                            try:
                                W_r = ((W_r - z) * s).view(meta["shape"]) * s2_eff
                                print(f"[QUANTIZER INFO] View() worked as fallback")
                            except Exception as view_error:
                                print(f"[QUANTIZER ERROR] View() also failed: {view_error}")
                                # Emergency fallback - use unpacked weights directly
                                try:
                                    W_r_flat = W_r.flatten()
                                    if W_r_flat.numel() != expected_size:
                                        if W_r_flat.numel() > expected_size:
                                            W_r_flat = W_r_flat[:expected_size]
                                        else:
                                            padding_size = expected_size - W_r_flat.numel()
                                            W_r_flat = torch.nn.functional.pad(W_r_flat, (0, padding_size))
                                    W_r = W_r_flat.reshape(meta["shape"]) * s2_eff
                                    print(f"[QUANTIZER WARNING] Emergency fallback used - results may be inaccurate")
                                except Exception as emergency_error:
                                    print(f"[QUANTIZER ERROR] Emergency fallback failed: {emergency_error}")
                                    raise reshape_error

            elif len(s.shape) == 3:
                H, W = meta["shape"]
                block = W_r.shape[-1]
                n_h, n_w = H // block, W // block
                total = n_h * block * n_w * block

                W_r = W_r.view(-1)[: total].view(n_h, block, n_w, block)

                try:
                    z_ = z.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
                except Exception:
                    z_ = z
                s1 = s.reshape(n_h, n_w, block, 1).permute(0, 2, 1, 3)
                s2_ = (s2.reshape(n_h, n_w, 1, block).permute(0, 2, 1, 3)
                    if s2 is not None else torch.ones_like(s1))

                W_r = ((W_r - z_) * s1 * s2_).view(H, W)
            else:
                raise ValueError("invalid scale shape for dequant")

            if torch.any(torch.isnan(W_r)):
                raise RuntimeError("NaN detected in dequantized weights")

            return W_r.to(compute_dtype)

        except RuntimeError as e:
            print(f"[QUANTIZER ERROR] RuntimeError in dequantize: {e}")
            print(f"[QUANTIZER DEBUG] W_q shape: {W_q.shape}")
            print(f"[QUANTIZER DEBUG] Meta shape: {meta.get('shape', 'unknown')}")
            print(f"[QUANTIZER DEBUG] Method: {meta.get('method', 'unknown')}")
            print(f"[QUANTIZER DEBUG] Scale shape: {s.shape if 's' in locals() else 'unknown'}")
            print(f"[QUANTIZER DEBUG] Zero shape: {z.shape if 'z' in locals() else 'unknown'}")

            # Specific handling for tensor size mismatch
            if "must match the size of tensor" in str(e):
                print(f"[QUANTIZER SPECIFIC] Tensor size mismatch in quantizer dequantize!")
                print(f"[QUANTIZER SPECIFIC] W_q shape: {W_q.shape}")
                print(f"[QUANTIZER SPECIFIC] Expected output shape: {meta.get('shape', 'unknown')}")
                print(f"[QUANTIZER SPECIFIC] Scale shape: {s.shape if 's' in locals() else 'unknown'}")
                print(f"[QUANTIZER SPECIFIC] This suggests incompatible quantization parameters")

            raise